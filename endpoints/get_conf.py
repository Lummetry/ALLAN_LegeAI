# -*- coding: utf-8 -*-

from libraries.model_server_v2 import FlaskWorker

import constants as ct
import spacy
import re
import phonenumbers
from string import punctuation
from utils.utils import simple_levenshtein_distance


_CONFIG = {
  'SPACY_MODEL' : 'ro_core_news_md',
 }


# CNP 
CNP_REG1 = re.compile(r'[0-9]{13}')
CNP_REG2 = re.compile(r'[1-8][0-9]{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[1-2][0-9]|3[0-1])[0-9]{6}')
CNP_NO_CHECK = 1
CNP_BASIC_CHECK = 2
CNP_FULL_CHECK = 3

# NER
MATCHED_NERS = {'PERSON', 'LOC'}

# NAME
PERSON_UPPERCASE = 1
PERSON_PROPN = 2
PERSON_TWO_WORDS = 3
MAX_LEV_DIST = 3

# ADDRESS
MIN_LOC_LENGTH = 10
ADDRESS_HAS_NUMBER = 0
ADDRESS_MIN_TOKENS = 3

# EMAIL
EMAIL_REG = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

# TELEPHONE
REG_START = r'(?:(?<!\d|\w)|(?<=^))'
REG_END = r'(?:(?=$)|(?!\d|\w))'
PHONE_REGS = {
    # 0722215678
    r'\d{4}[ .-]?\d{3}[ .-]?\d{3}',
    # +40789198780   
    r'\(?\+\d{2}\)?[ .-]?\d{3}[ .-]?\d{3}[ .-]?\d{3}',    
    # 6668945
    r'\d{3}[ .-]?\d{2}[ .-]?\d{2}',    
    # 0216668945
    r'\(?\d{3}\)?[ .-]?\d{3}[ .-]?\d{2}[ .-]?\d{2}'    
}
PHONE_REGS = [REG_START + r + REG_END for r in PHONE_REGS]
ALL_PHONE_REGS = '|'.join(PHONE_REGS)
PHONE_REG_CHECK = 0
PHONE_VALIDATION = 1
PHONE_REG_VALID = 2


class GetConfWorker(FlaskWorker):
    """
    Implementation of the worker for GET_CONFIDENTIAL endpoint
    """
    
    
    def __init__(self, **kwargs):
      super(GetConfWorker, self).__init__(**kwargs)
      return

    def _load_model(self):
    
        # Load Romanian spaCy dataset
        try:
            self.nlp_model = spacy.load(self.config_worker['SPACY_MODEL'])
        except OSError:
            spacy.cli.download(self.config_worker['SPACY_MODEL'])
            self.nlp_model = spacy.load(self.config_worker['SPACY_MODEL'])   
        self._create_notification('LOAD', 'Loaded spaCy model')
            
            
        return
    
    #######
    # AUX #
    #######
    
    def find_match(self, match, text, res):
        """ Find the position of a match """
        
        start = text.find(match)
        
        # If the same match already exists, look for the next one
        while start in res:
            start = text.find(match, start + 1)
            
        end = start + len(match)
                            
        return start, end
    
    def check_cnp(self, cnp):
        """ Check for a well formed CNP """
        
        cnp_const = '279146358279'
        check_digit = int(cnp[-1])
        
        s = 0
        for i, d in enumerate(cnp[:-1]):
            cnp_digit = int(d)
            const_digit = int(cnp_const[i])
            s += cnp_digit * const_digit
            
        r = s % 11
        
        if ((r < 10 and r == check_digit) or 
            (r == 10 and check_digit == 1)):
            return True
        
        return False
    
    
    def match_cnp(self, text, 
                  check_strength=CNP_FULL_CHECK
                ):
        """ Return the position of all the matches for CNP in a text. """
        
        # Find all matches
        if check_strength == CNP_NO_CHECK:
            cnp_reg = CNP_REG1
        else:
            cnp_reg = CNP_REG2
        
        matches = re.findall(cnp_reg, text)
        
        res = {}
        for match in matches:
            
            if check_strength < CNP_FULL_CHECK or (check_strength == CNP_FULL_CHECK and self.check_cnp(match)):
                start, end = self.find_match(match, text, res)
                res[start] = [start, end, 'CNP']
                if self.debug:
                    print(match)
                
        return res
    
    def next_name_code(self, code):
        """ Get the next code for a new name. """
        
        
        if code[0] == 'Z':
            next_code = 'A' * (len(code) + 1)
        else:
            next_code = [chr(ord(c) + 1) for c in code]
            next_code = ''.join(next_code)
            
        return next_code
    
    def find_name(self, name, person_dict):
        """ Find a name in the dictionary of names. """
        
        low_name = name.lower()
        
        for (n, code) in person_dict.items():
            lev_dist = simple_levenshtein_distance(n.lower(), low_name, 
                                                   normalize=False)
            if n.lower() == low_name or lev_dist < MAX_LEV_DIST:
                return code
            
        return None
    

    def check_token_condition(self, token, condition):
        """ Check if a token respects a condition """
        
        if ((condition == 'capital' and token.text[0].isupper())
            or (condition == 'propn' and token.pos_ == 'PROPN')):
            return True
        
        return False
    
    
    def check_name_condition(self, ent, doc, start_char, end_char,
                             condition
                            ):
        """ Only keep a sequence of words starting with a capital letter """
        
        is_match = False
                    
        i = ent.start
        while doc[i].idx < start_char:
            i += 1
        
        start = start_char
        end = end_char
                
        while doc[i].idx < end_char:
            token = doc[i]
                        
            if is_match == False:
                # Check one of the conditions
                if self.check_token_condition(token, condition):
                    start = token.idx
                    end = token.idx + len(token.text)
                    is_match = True
            else:
                # Check one of the conditions
                if not self.check_token_condition(token, condition):
                    break
                else:          
                    end = token.idx + len(token.text)
                        
            i += 1
            
        return is_match, start, end        
    
    def match_ner(self, nlp, text,
                  ners=MATCHED_NERS,
                  person_checks=[],
                  address_checks=[]
                 ):
        """ Return the position of spaCy named entities in a text. """
        matches = {}    
        
        if type(person_checks) == int:
            person_checks = [person_checks]
        
        doc = nlp(text)
        
        person_dict = {}
        current_code = 'A'
        
        for ent in doc.ents:
            
            if ent.label_ == 'PERSON':
                is_match = True
                
                start_char = ent.start_char
                end_char = ent.end_char
                    
                # Check POS
                if is_match and PERSON_PROPN in person_checks:
                    is_match, start_char, end_char = self.check_name_condition(ent, doc, 
                                                                               start_char, end_char,
                                                                               condition='propn')
                    
                # Check capital letters
                if is_match and PERSON_UPPERCASE in person_checks:
                    is_match, start_char, end_char = self.check_name_condition(ent, doc, 
                                                                               start_char, end_char,
                                                                               condition='capital')
                    
                # Add capitalized words to the right
                if end_char == ent.end_char:
                    idx = ent[-1].i + 1
                    
                    while idx < len(doc) and doc[idx].text[0].isupper():
                        end_char = doc[idx].idx + len(doc[idx])
                        idx += 1
                    
                # Check number of words
                if is_match and PERSON_TWO_WORDS in person_checks:
                    ent_text = text[start_char:end_char]
                    words = re.split("[" + punctuation + " ]+", ent_text)
                    if len(words) < 2:
                        is_match = False
                
                if is_match:
                            
                    # Ignore leading and trailing punctuation
                    while text[start_char] in punctuation:
                        start_char += 1
                    while text[end_char - 1] in punctuation:
                        end_char -= 1
                    
                    matches[start_char] = [start_char, end_char, "NUME"]
                    
                    person = text[start_char:end_char]
                    if self.debug:
                        print(person)
                                           
                    person_code = self.find_name(person, person_dict)
                    if not person_code:
                        # Get the next code for names
                        person_code = current_code
                        current_code = self.next_name_code(current_code)
                        
                    # Add the name to the dictionary
                    person_dict[person] = person_code
                        
                
            elif ent.label_ == 'LOC':
                
                if len(ent.text) > MIN_LOC_LENGTH and (ent.end - ent.start) > ADDRESS_MIN_TOKENS:
                    is_match = True
                    
                    if ADDRESS_HAS_NUMBER in address_checks:
                        is_match = False
                        for token in doc[ent.start : ent.end]:
                            if token.is_digit:
                                is_match = True
                                break
                    
                    if is_match:
                        matches[ent.start_char] = [ent.start_char, ent.end_char, "ADRESA"]
                        if self.debug:
                            print(ent)
                
        return matches, person_dict
    
    def match_email(self, text):
        """ Return the position of all the matches for email in a text. """
        
        matches = re.findall(EMAIL_REG, text)
        
        res = {}
        for match in matches:  
            start, end = self.find_match(match, text, res)
            res[start] = [start, end, 'EMAIL']
            if self.debug:
                print(match)
                
        return res
    
    def match_phone(self, text,
                    check_strength=PHONE_REG_CHECK
                   ):
        """ Return the position of all the matches for a phone number in a text. """
        
        res = {}
        
        if check_strength == PHONE_REG_CHECK or check_strength == PHONE_REG_VALID:
            matches = re.findall(ALL_PHONE_REGS, text)
    
            for match in matches:
                valid_match = True
                
                if check_strength == PHONE_REG_VALID:
                    checkMatch = phonenumbers.PhoneNumberMatcher(match, "RO")
                    if not checkMatch.has_next():  
                        valid_match = False
                        
                if valid_match:        
                    start, end = self.find_match(match, text, res)
                    res[start] = [start, end, 'TELEFON']
                    if self.debug:
                        print(match)
                
        elif check_strength == PHONE_VALIDATION:
            matches = phonenumbers.PhoneNumberMatcher(text, "RO")
    
            for match in matches:
                start, end = self.find_match(match, text, res)
                res[start] = [start, end, 'TELEFON']
                if self.debug:
                    print(match)
            
        return res
    
    #######
    # AUX #
    #######
    
    
    
    
    def _pre_process(self, inputs):
                
        doc = inputs['DOCUMENT']
        if len(doc) < ct.MODELS.TAG_MIN_INPUT:
          raise ValueError("Document: '{}' is below the minimum of {} words".format(
            doc, ct.MODELS.TAG_MIN_INPUT))
          
        self.debug = bool(inputs.get('DEBUG', False))
    
        return doc

    def _predict(self, prep_inputs):
        
        doc = prep_inputs    
        
        matches = {}
        
        # Match CNPS
        matches.update(self.match_cnp(doc))
    
        # Match email
        matches.update(self.match_email(doc))
        
        # Match address and name
        ner_matches, person_dict = self.match_ner(self.nlp_model, doc, person_checks=[PERSON_PROPN, PERSON_UPPERCASE, PERSON_TWO_WORDS])
        matches.update(ner_matches)  
        if self.debug:
            print(person_dict)

        # Match phone
        matches.update(self.match_phone(doc, check_strength=PHONE_REG_VALID))
              
        return doc, matches, person_dict

    def _post_process(self, pred):
        
        doc, matches, person_dict = pred
        
        # Order matches 
        match_tuples = list(matches.values())
        match_starts = list(sorted(matches.keys(), reverse=True))
    
        # Replace all confidential information (except names) in text
        hidden_doc = doc
        for key in match_starts:
            [start, end, label] = matches[key]
            if label != 'NUME':
                hidden_doc = hidden_doc[:start] + '***' + hidden_doc[end:]
        
        # Replace names with their codes
        for (name, code) in person_dict.items():
            
            # Search for all occurances of name
            while True:
                # Ignore the case of the letters
                name_match = re.search(name.lower(), hidden_doc.lower())
                
                if name_match:
                    start, end = name_match.span()
                    hidden_doc = hidden_doc[:start] + code + hidden_doc[end:]
                else:
                    break
            
        res = {}
        res['positions'] = match_tuples
        res['output'] = hidden_doc
        
        if self.debug:
            print(hidden_doc)
        
        return res


if __name__ == '__main__':
  from libraries import Logger

  l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
  eng = GetConfWorker(log=l, default_config=_CONFIG, verbosity_level=1)
  
  test = {
      'DEBUG' : False,
      
           # 'DOCUMENT': """Un contribuabil al cărui cod numeric personal este 1520518054675 va completa caseta "Cod fiscal" astfel:""",
      
      # 'DOCUMENT': """Se desemnează domnul Cocea Radu, avocat, cocea@gmail.com, 0216667896 domiciliat în municipiul Bucureşti, Bd. Laminorului nr. 84, sectorul 1, legitimat cu C.I. seria RD nr. 040958, eliberată la data de 16 septembrie 1998 de Secţia 5 Poliţie Bucureşti, CNP 1561119034963, în calitate de administrator special. Se desemneaza si doamna Alice Munteanu cu telefon 0216654343, domiciliata in Bd. Timisoara nr. 107 """, 
        
      # 'DOCUMENT': """Cod numeric personal: 1505952103022. Doi copii minori înregistraţi în documentul de identitate.""",
        
      # 'DOCUMENT': """Bătrîn Cantemhir-Marian, porcine, Str. Cardos Iacob nr. 34, Arad, judeţul Arad, 1850810020101. 
      # Almăjanu Steliana, porcine, Comuna Peretu, judeţul Teleorman, 2580925341708.""",
      
      'DOCUMENT' : """
III. În baza art. 396 al. 1 şi 5 din Codul de procedură penală rap. la art. 16 al. 1 lit. b din Codul de procedură penală a fost achitat inculpatul MIHALACHE GABRIEL-CONSTANTIN, fiul lui Marin şi Marioara - Aurora, născut la 18.05.1952 în Brad, jud. Hunedoara, domiciliat în Oradea, strada Episcop Ioan Suciu nr.4, bloc ZP2, apt.10, CNP 1520518054675, pentru săvârşirea infracţiunii de efectuarea unei prelevări atunci când prin aceasta se compromite o autopsie medico-legală, prev. de art. 155 din Legea nr. 95/2006 republicată.
	În baza art. 397 al. 1 din Codul de procedură penală s-a luat act că persoanele vătămate Lozincă Maria, Ministerul Public – Parchetul de pe lângă Înalta Curte de Casaţie şi Justiţie şi Parchetul de pe lângă Tribunalul Bihor nu 
s-au constituit părţi civile în cauză.
	În baza art. 274 al. 1 din Codul de procedură penală a fost obligat inculpatul Popa Vasile Constantin la plata sumei de 20.000 lei cu titlu de cheltuieli judiciare faţă de stat.
	În baza art. 275 al. 3 din Codul de procedură penală cheltuielile judiciare ocazionate cu soluţionarea cauzei faţă de inculpaţii David Florian Alin şi Mihalache Gabriel Constantin, au rămas în sarcina statului.
	În baza art. 275 al. 6 din Codul de procedură penală s-a dispus plata din fondurile Ministerului Justiţiei către Baroul Timiş a sumei de câte 350 lei, reprezentând onorariu parţial avocat din oficiu către avocaţii Schiriac Lăcrămioara şi Miloş Raluca, respectiv 100 lei, reprezentând onorariu parţial avocat din oficiu către avocatul Murgu Călin.
	În baza art. 120 al. 2 lit. a teza 2 din Codul de procedură penală a fost respinsă cererea de acordare a cheltuielilor de transport pentru termenul din 2 noiembrie 2016, formulată de martora Bodin Alina Adriana.
	În baza art. 120 al. 2 lit. a teza 2 din Codul de procedură penală a fost admisă în parte cererea formulată de martorul Popa Vasile cu privire la acordarea cheltuielilor legate de deplasarea la instanţă. 
S-a dispus plata din fondurile Ministerului Justiţiei a sumei de 480 lei, reprezentând contravaloarea serviciilor de cazare privind pe martorul Popa Vasile, pentru termenele de judecată din 31 octombrie 2017 şi 7 noiembrie 2017 şi respinge în rest cererea formulată.
Pentru a pronunţa această sentinţă, prima instanţă a reţinut că prin rechizitoriul nr. 421/P/2013 din 17.12.2014 al Parchetului de pe lângă Înalta Curte de Casaţie şi Justiţie – Direcţia Naţională Anticorupţie, înregistrat la Curtea de Apel Oradea sub nr. dosar 490/35/2014 la data de 19.12.2014, s-a dispus trimiterea în judecată a inculpaţilor POPA VASILE CONSTANTIN, pentru săvârşirea infracţiunii de luare de mită, prev. de art. 6 din L. 78/2000 rap. la art. 289 din Codul penal  rap. la art. 7 al. 1 lit. b din L. 78/2000; infracţiunii de şantaj, prev. de art. 207 al. 1 din Codul penal, cu aplic. art. 13 ind. 1 din L. 78/2000; 3 infracţiuni de abuz în serviciu, prev. de art. 13 ind. 2 din L. 78/2000 cu referire la art. 297 al. 1 din Codul penal; 3 infracţiuni de distrugere de înscrisuri prev. de art. 242 al. 1 şi 3 din Codul penal; 3 infracţiuni de favorizare a făptuitorului, prev. de art. 269 al. 1 din Codul penal; infracţiunii de instigare la efectuarea unor prelevări atunci când prin aceasta se compromite o autopsie medico-legală, prev. de art. 47 din Codul penal rap. la art. 156 din Legea nr. 95/2006 şi a infracţiunii de trafic de influenţă prev. de art. 6 din Legea nr. 78/2000 rap. la art. 291 din Codul penal rap. la art. 7 al. 1 lit. b din Legea nr. 78/2000; MIHALACHE GABRIEL CONSTANTIN, pentru săvârşirea infracţiunii de efectuare a unei prelevări atunci când prin aceasta se compromite o autopsie medico-legală, prev. de art. 156 din Legea nr. 95/2006 republicată şi DAVID FLORIAN ALIN, pentru săvârşirea infracţiunii de cumpărare de influenţă, prev. de art. 6 din Legea nr. 78/2000 rap. la art. 292 din Codul penal, cu aplicarea art. 5 din Codul penal. 

"""
        
      }
  
  res = eng.execute(inputs=test, counter=1)
  print(res)
