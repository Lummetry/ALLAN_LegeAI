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
        print(doc.ents)
        
        person_dict = {}
        current_code = 'A'
        
        for ent in doc.ents:
            
            if ent.label_ == 'PERSON':
                is_match = True
                
                start_char = ent.start_char
                end_char = ent.end_char
                
                if PERSON_UPPERCASE in person_checks and not ent.text.istitle():
                    # Check if all words start with an uppercase letter
                    is_match = False
                    
                if is_match and PERSON_TWO_WORDS in person_checks and len(ent) <= 1:
                    is_match = False
                    
                if is_match and PERSON_PROPN in person_checks:
                    is_match = False
                    
                    # Check if all words are proper nouns
                    for token in doc[ent.start : ent.end]:
                        if is_match == False:
                            if token.pos_ == 'PROPN':
                                start_char = token.idx
                                is_match = True
                        else:
                            if token.pos_ != 'PROPN':
                                end_char = token.idx + len(token.text)
                                break
                            
                # Ignore leading and trailing punctuation
                while text[start_char] in punctuation:
                    start_char += 1
                while text[end_char - 1] in punctuation:
                    end_char -= 1
                
                if is_match:
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
        self.print_text = bool(inputs.get('PRINT_TEXT', False))
    
        return doc

    def _predict(self, prep_inputs):
        
        doc = prep_inputs    
        
        matches = {}
        
        # Match CNPS
        matches.update(self.match_cnp(doc))
    
        # Match email
        matches.update(self.match_email(doc))
        
        # Match address and name
        ner_matches, person_dict = self.match_ner(self.nlp_model, doc, person_checks=[PERSON_PROPN, PERSON_TWO_WORDS])
        matches.update(ner_matches)  
        print(person_dict)

        # Match phone
        matches.update(self.match_phone(doc, check_strength=PHONE_REG_VALID))
              
        return doc, matches, person_dict

    def _post_process(self, pred):
        
        doc, matches, person_dict = pred
        print(matches)
        
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
            hidden_doc = hidden_doc.replace(name, code)
        
        res = {}
        res['positions'] = match_tuples
        res['output'] = hidden_doc
        
        # if self.debug:
        #     print(hidden_doc)
        
        return res


if __name__ == '__main__':
  from libraries import Logger

  l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
  eng = GetConfWorker(log=l, default_config=_CONFIG, verbosity_level=1)
  
  test = {
      'DEBUG' : True,
      'PRINT_RESULT': True,
      
           # 'DOCUMENT': """Un contribuabil al cărui cod numeric personal este 1520518054675 va completa caseta "Cod fiscal" astfel:""",
      
      # 'DOCUMENT': """Se desemnează domnul Cocea Radu, avocat, cocea@gmail.com, 0216667896 domiciliat în municipiul Bucureşti, Bd. Laminorului nr. 84, sectorul 1, legitimat cu C.I. seria RD nr. 040958, eliberată la data de 16 septembrie 1998 de Secţia 5 Poliţie Bucureşti, CNP 1561119034963, în calitate de administrator special. Se desemneaza si doamna Alice Munteanu cu telefon 0216654343, domiciliata in Bd. Timisoara nr. 107 """, 
        
      # 'DOCUMENT': """Cod numeric personal: 1505952103022. Doi copii minori înregistraţi în documentul de identitate.""",
        
      # 'DOCUMENT': """Bătrîn Cantemhir-Marian, porcine, Str. Cardos Iacob nr. 34, Arad, judeţul Arad, 1850810020101. 
      # Almăjanu Steliana, porcine, Comuna Peretu, judeţul Teleorman, 2580925341708.""",
      
      'DOCUMENT' : """
      S-au luat în examinare apelurile declarate de Parchetul de pe lângă Înalta Curte de Casaţie şi Justiţie – Direcţia naţională Anticorupţie şi de inculpatul Popa Vasile Constantin împotriva sentinţei penale nr. 194/PI din 13 martie 2018 a Curţii de Apel Timişoara – Secţia Penală.
Dezbaterile au fost consemnate în încheierea de şedinţă din data de 09 ianuarie 2020,  ce face parte integrantă din prezenta decizie şi având nevoie de timp pentru a delibera, în baza art. 391 din Codul de procedură penală a amânat pronunţarea pentru azi 22 ianuarie 2020, când în aceeaşi compunere a pronunţat următoarea decizie:
ÎNALA CURTE
	Asupra apelurilor penale de faţă;
În baza lucrărilor din dosar, constată următoarele:
Prin sentinţa penală nr. 194/PI din 13 martie 2018 a Curţii de Apel Timişoara – Secţia Penală, pronunţată în dosarul nr.490/35/2014, în baza art. 386 din Codul de procedură penală a respins cererea de schimbare a încadrării juridice a faptei de sustragere sau distrugere de înscrisuri, prev. de art. 242 al. 1 şi 3 din Codul penal, cu aplic. art. 5 din Codul penal, în cea de sustragere sau distrugere de probe ori de înscrisuri, prev. de art. 275 al. 1 şi 2 din Codul penal, formulată de inculpatul POPA VASILE CONSTANTIN.
	În baza art. 386 din Codul de procedură penală a fost admisă cererea de schimbare a încadrării juridice a faptelor formulată de D.N.A. – Serviciul Teritorial Timişoara, sens în care:
- s-a dispus schimbarea încadrării juridice a infracţiunii de luare de mită, prev. de art. 6 din Legea nr. 78/2000 rap. la art. 289 din Codul penal, rap. la art. 7 alin.1 lit. b din Legea nr. 78/2000, cu aplic. art. 5 din Codul penal, în infracţiunea de luare de mită în formă continuată, prev. de art. 6 din Legea nr. 78/2000 rap. la art. 289 al. 1 din Codul penal, rap. la art. 7 alin.1 lit. b din Legea nr. 78/2000, cu aplic. art. 35 al. 1 din Codul penal şi art. 5 din Codul penal (5 acte materiale), cu privire la inculpatul POPA VASILE CONSTANTIN; 
- s-a dispus schimbarea încadrării juridice a infracţiunii de trafic de influenţă prev. de art. 6 din Legea nr. 78/2000  rap. la art. 291 din Codul penal., rap. la art. 7 alin.1, lit. b din Legea nr. 78/2000, cu aplic. art. 5 din Codul penal, în infracţiunea de trafic de influenţă prev. de art. 6 din Legea nr. 78/2000  rap. la art. 291 al. 1 din Codul penal, rap. la art. 7 alin.1, lit. b din Legea nr. 78/2000, cu aplic. art. 5 din Codul penal, cu privire la inculpatul POPA VASILE CONSTANTIN; 
- s-a dispus schimbarea încadrării juridice a infracţiunii de sustragere sau distrugere de înscrisuri, prev. de art. 242 alin. 1 şi 3 din Codul penal, cu aplic. art. 5 din Codul penal, în infracţiunea de sustragere sau distrugere de înscrisuri prev. de art. 259 al. 1 şi 2 din Codul penal, cu aplic. art. 5 din Codul penal (3 infracţiuni), cu privire la inculpatul POPA VASILE CONSTANTIN; 
- s-a dispus schimbarea încadrării juridice a infracţiunii de instigare la efectuarea unei prelevări atunci când prin aceasta se compromite o autopsie medico-legală, prev. de art. 47 din Codul penal rap. la art. 156 din Legea nr. 95/2006 republicată, în infracţiunea de instigare la efectuarea unei prelevări atunci când prin aceasta se compromite o autopsie medico-legală, prev. de art. 47 din Codul penal rap. la art. 155 din Legea nr. 95/2006 republicată, cu privire la inculpatul POPA VASILE CONSTANTIN; 
- s-a dispus schimbarea încadrării juridice a infracţiunii de cumpărare de influenţă, prev. de art. 6 din Legea nr. 78/2000 rap. la art. 292 Codul penal, cu aplic. art. 5 din Codul penal, în infracţiunea de cumpărare de influenţă, prev. de art. 6 din Legea nr. 78/2000 rap. la art. 292 al. 1 din Codul penal, cu aplic. art. 5 din Codul penal, cu privire la inculpatul DAVID FLORIAN ALIN; 
- s-a dispus schimbarea încadrării juridice a infracţiunii de efectuarea unei prelevări atunci când prin aceasta se compromite o autopsie medico-legală, prev. de art. 156 din Legea nr. 95/2006 republicată, în infracţiunea de efectuarea unei prelevări atunci când prin aceasta se compromite o autopsie medico-legală, prev. de art. 155 din Legea nr. 95/2006 republicată, cu privire la inculpatul MIHALACHE GABRIEL CONSTANTIN.
	I. În baza art. 396 al. 1 şi 5 din Codul de procedură penală rap. la art. 16 al. 1 lit. b din Codul de procedură penală a fost achitat inculpatul POPA VASILE CONTASNTIN, sub aspectul săvârşirii infracţiunii de luare de mită în formă continuată, prev. de art. 6 din Legea nr. 78/2000 rap. la art. 289 al. 1 din Codul penal, rap. la art. 7 alin.1 lit. b din Legea nr. 78/2000, cu aplic. art. 35 al. 1 Codul penal şi art. 5 din Codul penal (5 acte materiale).
În baza art. 396 al. 1 şi 5 din Codul de procedură penală rap. la art. 16 al. 1 lit. a din Codul de procedură penală a fost achitat inculpatul POPA VASILE CONSTANTIN, pentru săvârşirea infracţiunii de abuz în serviciu, prev. de art. 132 din Legea nr. 78/2000 modificată, cu referire la art. 297 alin. 1 din Codul 


penal, cu aplic. art. 5 din Codul penal în legătură cu dosarul penal nr. 7157/P/2008 al Parchetului de pe lângă Judecătoria Oradea.
"""
        
      }
  
  res = eng.execute(inputs=test, counter=1)
  print(res)
