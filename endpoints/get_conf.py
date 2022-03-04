# -*- coding: utf-8 -*-

from libraries.model_server_v2 import FlaskWorker

import constants as ct
import spacy
from spacy.matcher import DependencyMatcher
import re
import phonenumbers
from string import punctuation
from utils.utils import simple_levenshtein_distance
import unidecode
import json


_CONFIG = {
  'SPACY_MODEL' : 'ro_core_news_lg',
  'INSTITUTION_LIST' : 'C:\\Proiecte\\LegeAI\\ALLAN_LegeAI\\_cache\\_data\\nomenclator_institutii_publice.txt',
  'CONF_REGEX' : 'C:\\Proiecte\\LegeAI\\Date\\Task6\\conf_regex.json'
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
ADDRESS_MIN_TOKENS = 1
ADDRESS_MERGE_DIST = 3
ADDRESS_INCLUDE_GPE = 4
ADDRESS_REMOVE_PUNCT = 5

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

# SERIE NUMAR CI
SERIE_NUMAR_REGS = {
    r'seri(?:e|a) [A-Z]{2}.{0,5}num[aă]r \d{6}',
    r'seri(?:e|a) [A-Z]{2}.{0,5}nr(?:.)? \d{6}',
    r'seri(?:e|a) [A-Z]{2}\d{6}',
    r'num[aă]r [A-Z]{2}\d{6}',
    r'nr(?:.)? [A-Z]{2}\d{6}'     
}
REG_END = r'(?:(?=$)|(?!\d|\w))'
SERIE_NUMAR_REGS = [r + REG_END for r in SERIE_NUMAR_REGS]
ALL_SERIE_NUMAR_REGS = '|'.join(SERIE_NUMAR_REGS)
SERII = ["AX", "TR", "AR", "ZR", "XC", "ZC", "MM", "XM", "XB", "XT", "BV", "ZV", "XR", "DP", "DR", "DT", "DX", "RD", 
         "RR", "RT", "RX", "RK", "IF", "XZ", "KL", "KX", "CJ", "KT", "KZ", "DX", "DZ", "HD", "VN", "GL", "ZL", "GG", 
         "MX", "MZ", "IZ", "MH", "HR", "XH", "ZH", "NT", "AS", "AZ", "PH", "PX", "KS", "VX", "SM", "KV", "SB", "OT", 
         "SZ", "SV", "XV", "TM", "TZ", "DD", "GZ", "ZS", "MS", "TC", "VS", "SX"]
SERIE_CHECK = 1

# IBAN
IBAN_REG = r'RO[ ]?\d{2}[ ]?\w{4}(?:[ ]?[A-Z0-9]{4}){4}'
IBAN_REG = REG_START + IBAN_REG + REG_END

# CUI
CUI_REGS = {
    # CUI 278973
    r'(?:CUI|CIF)(?: )?(?:RO)?\d{2,10}',
    
    # RO278973
    r'RO\d{2,10}',    
    
    # J12/123456/2000
    r'(?:J|F|C)(?: )?\d{1,2}\/\d{1,7}\/\d{4}',
} 
CUI_REGS = [REG_START + r + REG_END for r in CUI_REGS]
ALL_CUI_REGS = '|'.join(CUI_REGS)

# BRAND
BRAND_INCLUDE_FACILITY = 1
BRAND_EXCLUDE_COMMON = 2

# REGISTRY
REGISTRY_REG = '([^0-9A-Z:-_/|]{0,10})([0-9A-Z:-_/| ]{3,})'

# EU CASE
EU_CASE_REGS = {
    # C-XXX/XX
    r'(?:C|T)-\d{2,3}\/\d{2}',
    
} 
EU_CASE_REGS = [REG_START + r + REG_END for r in EU_CASE_REGS]
ALL_EU_CASE_REGS = '|'.join(EU_CASE_REGS)
MIN_CASE_DISTANCE = 20


class GetConfWorker(FlaskWorker):
    """
    Implementation of the worker for GET_CONFIDENTIAL endpoint
    """
    
    
    def __init__(self, **kwargs):
      super(GetConfWorker, self).__init__(**kwargs)
      return

    def _load_model(self):
        
        # Read list of public institutions
        inst_file = open(self.config_worker['INSTITUTION_LIST'], 'r', encoding='utf-8')
        self.institution_list = inst_file.read().splitlines()
        
        # Read JSON REGEX file
        json_file = open(self.config_worker['CONF_REGEX'], 'r', encoding="utf-8")
        json_string = json_file.read().replace('\\', '\\\\')
        json_data = json.loads(json_string)
        
        self.registry_keywords = '|'.join(json_data['registry_dict'])
        self.conf_regex_list = json_data['conf_regex']
    
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
    
    def match_serie_numar(self, text, serie_checks=[]):
        
        matches = re.findall(ALL_SERIE_NUMAR_REGS, text)
        
        res = {}
        for match in matches:
                
            if SERIE_CHECK in serie_checks:
                found_serie = False
                for serie in SERII:
                    if serie in match:
                        found_serie = True
                
            if len(serie_checks) == 0 or (SERIE_CHECK in serie_checks and
                                          found_serie == True):
                start, end = self.find_match(match, text, res)
                res[start] = [start, end, 'SERIENUMAR']
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
    
    
    def check_name_condition(self, matches, doc, condition):
        """ Split matches in sequence of words which respect a condition. """
        
        new_matches = []
        
        for (start, end) in matches:
    
            current_match_start = -1
            current_match_end = -1
    
            for i in range(start, end):
                token = doc[i]
    
                if self.check_token_condition(token, condition):
                    if current_match_start == -1:
                        current_match_start = i
                    current_match_end = i + 1
    
                if not self.check_token_condition(token, condition) or i == end - 1:
                    if current_match_start != -1:
    
                        new_matches.append((current_match_start,current_match_end))
    
                        current_match_end = -1
                        current_match_start = -1
                    
        return new_matches  
    
    def match_name(self, nlp, doc, text,
                   person_checks=[]
                  ):
        """ Return the position of names in a text. """  
        
        if type(person_checks) == int:
            person_checks = [person_checks]
        
        person_dict = {}
        current_code = 'A'
        
        # Form the list of candidate matches
        candidate_matches = []
        for ent in doc.ents:
            if ent.label_ == 'PERSON':
                candidate_matches.append((ent.start, ent.end))
                    
        # Check POS PROPN condition
        if PERSON_PROPN in person_checks:
            candidate_matches = self.check_name_condition(candidate_matches, doc, condition='propn')
    
        # Check uppercase words
        if PERSON_UPPERCASE in person_checks:
            candidate_matches = self.check_name_condition(candidate_matches, doc, condition='capital')
                
        # Check other conditions
        final_matches = {}
        for (start, end) in candidate_matches:
            
            # Add capitalized words to the right    
            new_end = end
            while new_end < len(doc) and doc[new_end].text[0].isupper():
                # While next word starts with capital letter
                new_end += 1
                
            # Add capitalized words to the left    
            new_start = start
            while new_start > 0 and doc[new_start - 1].text[0].isupper():
                # While previous word starts with capital letter
                new_start -= 1
            
            # Check minimum number of words
            is_match = True
            if PERSON_TWO_WORDS in person_checks:
                ent_text = doc[new_start : new_end].text
                words = re.split("[" + punctuation + " ]+", ent_text)
                if len(words) < 2:
                    is_match = False
                    
                    
            if is_match:
                start_idx = doc[new_start].idx
                end_idx = doc[new_end - 1].idx + len(doc[new_end - 1])
                
                # Ignore leading and trailing punctuation
                while text[start_idx] in punctuation:
                    start_idx += 1
                while text[end_idx - 1] in punctuation:
                    end_idx -= 1
                
                final_matches[start_idx] = [start_idx, end_idx, "NUME"]
                
                person = text[start_idx:end_idx]
                if self.debug:
                    print(person)
                                       
                person_code = self.find_name(person, person_dict)
                if not person_code:
                    # Get the next code for names
                    person_code = current_code
                    current_code = self.next_name_code(current_code)
                    
                # Add the name to the dictionary
                person_dict[person] = person_code
                
        return final_matches, person_dict
    
    # def match_name(self, nlp, doc, text,
    #                person_checks=[]
    #                ):
    #     """ Return the position of namess in a text. """
    #     matches = {}    
        
    #     if type(person_checks) == int:
    #         person_checks = [person_checks]
        
    #     person_dict = {}
    #     current_code = 'A'
        
    #     for ent in doc.ents:
    #         # print(ent, ent.label_)
            
    #         if ent.label_ == 'PERSON':
    #             is_match = True
                
    #             start_char = ent.start_char
    #             end_char = ent.end_char
                    
    #             # Check POS
    #             if is_match and PERSON_PROPN in person_checks:
    #                 is_match, start_char, end_char = self.check_name_condition(ent, doc, 
    #                                                                            start_char, end_char,
    #                                                                            condition='propn')
                    
    #             # Check capital letters
    #             if is_match and PERSON_UPPERCASE in person_checks:
    #                 is_match, start_char, end_char = self.check_name_condition(ent, doc, 
    #                                                                            start_char, end_char,
    #                                                                            condition='capital')
                    
    #             # Add capitalized words to the right
    #             if end_char == ent.end_char:
    #                 idx = ent[-1].i + 1
                    
    #                 while idx < len(doc) and doc[idx].text[0].isupper():
    #                     end_char = doc[idx].idx + len(doc[idx])
    #                     idx += 1
                    
    #             # Check number of words
    #             if is_match and PERSON_TWO_WORDS in person_checks:
    #                 ent_text = text[start_char:end_char]
    #                 words = re.split("[" + punctuation + " ]+", ent_text)
    #                 if len(words) < 2:
    #                     is_match = False
                
    #             if is_match:
                            
    #                 # Ignore leading and trailing punctuation
    #                 while text[start_char] in punctuation:
    #                     start_char += 1
    #                 while text[end_char - 1] in punctuation:
    #                     end_char -= 1
                    
    #                 matches[start_char] = [start_char, end_char, "NUME"]
                    
    #                 person = text[start_char:end_char]
    #                 if self.debug:
    #                     print(person)
                                           
    #                 person_code = self.find_name(person, person_dict)
    #                 if not person_code:
    #                     # Get the next code for names
    #                     person_code = current_code
    #                     current_code = self.next_name_code(current_code)
                        
    #                 # Add the name to the dictionary
    #                 person_dict[person] = person_code
            
                
    #     return matches, person_dict        
    
    def remove_punct_tokens(self, doc):
        ''' 
        Remove the punctuation tokens from a Doc.
        Returns the new text along with the positions of the new tokens in the original text.
        '''
        
        # Remove the punctuation
        token_pos_dict = {}
        new_text = []
    
        new_index = 0
        for d in doc:
            # Only keep non-punctuation tokens
            if not d.is_punct:
                new_text.append(d.text)
                # Keep track of the position of the token in the original text
                token_pos_dict[new_index] = (d.idx, d.idx + len(d))
                new_index += 1
        
        # Get the new text
        new_text = ' '.join(new_text)
        
        return new_text, token_pos_dict
    
    def get_entity_original_pos(self, e, token_pos_dict):
        ''' Get the start and end positions of an entity in the original text. '''
        
        (start, _) = token_pos_dict[e.start]
        (_, end) = token_pos_dict[e.end - 1]
        
        return start, end
    
    def check_loc_entities(self, doc, matches, address_checks,
                           token_pos_dict=None):
        ''' Check the LOC entities in a Doc. '''
        
        for ent in doc.ents:
               
            if ent.label_ == 'LOC':
                if len(ent.text) > MIN_LOC_LENGTH and (ent.end - ent.start) > ADDRESS_MIN_TOKENS:
                    is_match = True
    
                    if ADDRESS_HAS_NUMBER in address_checks:
                        is_match = False
                        for token in doc[ent.start : ent.end]:
                            if token.is_digit:
                                is_match = True
                                break
    
                    if is_match:
                        if token_pos_dict:
                            # If a position dictionary for the Tokens was given, make the changes
                            orig_start, orig_end = self.get_entity_original_pos(ent, token_pos_dict)
                        else:
                            orig_start, orig_end = ent.start_char, ent.end_char
    
                        # Check if it could be merged with nearby address
                        merged = False
    
                        for (s, m) in matches.items():
                            match_type = m[2]
                            e = m[1]
                            
                            if match_type == 'ADRESA' and (max(s, orig_start) - min(e, orig_end)) < ADDRESS_MERGE_DIST:
                                # If the matches are overlapping or are close enough
                                matches[min(s, orig_start)] = [min(s, orig_start), max(e, orig_end), "ADRESA"]
                                merged = True
                                break
    
                        if not merged:
                            matches[orig_start] = [orig_start, orig_end, "ADRESA"]
                        
                        if self.debug:
                            print(ent)
                        
        return matches
    
    def match_address(self, nlp, doc, text, 
                      address_checks=[ADDRESS_INCLUDE_GPE, ADDRESS_REMOVE_PUNCT]
                     ):
        """ Return the position of address entities in a text. """
        matches = {}    
        
        # Add all GPE entities
        if ADDRESS_INCLUDE_GPE in address_checks:
            for ent in doc.ents:
                if ent.label_ == 'GPE':
                    matches[ent.start_char] = [ent.start_char, ent.end_char, "ADRESA"]
                    if self.debug:
                        print(ent)
    
        # Check all LOC entities from initial Doc
        matches = self.check_loc_entities(doc, matches, address_checks)  
        
        # Remove punctuation and check LOC entities again
        if ADDRESS_REMOVE_PUNCT in address_checks:
            
            # Get the new text, with punctuation removed
            new_text, token_pos_dict = self.remove_punct_tokens(doc)
    
            # Build a new spaCy Doc
            doc_rp = nlp(new_text)
            
            matches = self.check_loc_entities(doc_rp, matches, address_checks,
                                              token_pos_dict=token_pos_dict)
                
        return matches
    
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
    
    def match_institution(self, nlp, text, public_insts):
        """ Return the position of all the matches for institutions in a text. """
        
        matches = {}    
        
        doc = nlp(text)
        
        # Collect all Organization matches
        for ent in doc.ents:
            
            if ent.label_ == 'ORGANIZATION':
                
                is_match = True
    
                start_char = ent.start_char
                end_char = ent.end_char
                
                # Filter matches the represent public institutions
                
                match_normalized = unidecode.unidecode(ent.text.lower())
                match_stripped = ''.join(filter(str.isalnum, match_normalized))
                
                for public_inst in public_insts:
    
                    if match_normalized == public_inst or match_stripped == public_inst:
                        is_match = False
                        break
    
                if is_match:
                    matches[start_char] = [start_char, end_char, "INSTITUTIE"]
                    if self.debug:
                        print(ent.text)
                
        return matches
    
    def match_iban(self, text):
        """ Return the position of all the matches for IBANs in a text. """
       
        matches = re.findall(IBAN_REG, text)
            
        res = {}
        for match in matches:
                
            start, end = self.find_match(match, text, res)
            res[start] = [start, end, 'CNP']
               
            if self.debug: 
                print(match)
        
        return res
    
    def birthdate_matcher(self):
        ''' Get the spaCy Dependency Matcher for birthdate'''
        
        matcher = DependencyMatcher(self.nlp_model.vocab)
        
        nascut_syn = ["nascut", "nascuta", "nascuti", "nascute", "naste",
                      "născut", "născută", "născuți", "născute", "naște"]
    
        nastere_syn = ["nasterii", "nașterii", "nastere", "naștere"]
    
        date_shape = ["dd.dd.dddd", "dd/dd/dddd", "dd-dd-dddd",
                      "dd.dd.dd", "dd/dd/dd", "dd-dd-dd",
                      "dddd.dd.dd", "dddd/dd/dd", "dddd-dd-dd",
                     ]
    
        birthdate_pattern1 = [
    
            # nascuta la data 20.05.1989
            [{"RIGHT_ID" : "anch_nascut", "RIGHT_ATTRS" : {"ORTH" : {"IN" : nascut_syn}}},
             {"LEFT_ID" : "anch_nascut", "REL_OP" : ">", "RIGHT_ID" : "anch_data", "RIGHT_ATTRS" : {"LEMMA" : "dată"}},
             {"LEFT_ID" : "anch_data", "REL_OP" : ">", "RIGHT_ID" : "data", "RIGHT_ATTRS" : {"DEP" : {"IN" : ["nummod", "nmod"]},
                                                                                             "SHAPE" : {"IN" : date_shape}}}
            ],
    
            # data nasterii 20.05.1989
            [{"RIGHT_ID" : "anch_data", "RIGHT_ATTRS" : {"LEMMA" : "dată"}},
             {"LEFT_ID" : "anch_data", "REL_OP" : ">", "RIGHT_ID" : "anch_nastere", "RIGHT_ATTRS" : {"ORTH" : {"IN" : nastere_syn}}},
             {"LEFT_ID" : "anch_nastere", "REL_OP" : ">", "RIGHT_ID" : "data", "RIGHT_ATTRS" : {"DEP" : {"IN" : ["nummod", "nmod"]},
                                                                                                "SHAPE" : {"IN" : date_shape}}}
            ],

            # data nastere 20.05.1989
            [{"RIGHT_ID" : "anch_data", "RIGHT_ATTRS" : {"LEMMA" : "dată"}},
             {"LEFT_ID" : "anch_data", "REL_OP" : ">", "RIGHT_ID" : "anch_nastere", "RIGHT_ATTRS" : {"ORTH" : {"IN" : nastere_syn}}},
             {"LEFT_ID" : "anch_data", "REL_OP" : ">", "RIGHT_ID" : "data", "RIGHT_ATTRS" : {"DEP" : {"IN" : ["nummod", "nmod", "obl"]},
                                                                                                "SHAPE" : {"IN" : date_shape}}}
            ],
        ]
        matcher.add("birthdate1", birthdate_pattern1)
    
        birthdate_pattern2 = [
    
            # nascuta la data 20 iunie 1989
            [{"RIGHT_ID" : "anch_nascut", "RIGHT_ATTRS" : {"ORTH" : {"IN" : nascut_syn}}},
             {"LEFT_ID" : "anch_nascut", "REL_OP" : ">", "RIGHT_ID" : "anch_data", "RIGHT_ATTRS" : {"LEMMA" : "dată"}},
             {"LEFT_ID" : "anch_data", "REL_OP" : ">", "RIGHT_ID" : "luna", "RIGHT_ATTRS" : {"DEP" : "nmod"}},
             {"LEFT_ID" : "luna", "REL_OP" : ";", "RIGHT_ID" : "zi", "RIGHT_ATTRS" : {"DEP" : "nummod"}},
             {"LEFT_ID" : "luna", "REL_OP" : ".", "RIGHT_ID" : "an", "RIGHT_ATTRS" : {"DEP" : "nummod"}}
            ],
    
            # data nasterii 20 mai 1989
            [{"RIGHT_ID" : "anch_data", "RIGHT_ATTRS" : {"LEMMA" : "dată"}},
             {"LEFT_ID" : "anch_data", "REL_OP" : ">", "RIGHT_ID" : "anch_nastere", "RIGHT_ATTRS" : {"ORTH" : {"IN" : nastere_syn}}},
             {"LEFT_ID" : "anch_nastere", "REL_OP" : ">", "RIGHT_ID" : "luna", "RIGHT_ATTRS" : {"DEP" : "nmod"}},
             {"LEFT_ID" : "luna", "REL_OP" : ";", "RIGHT_ID" : "zi", "RIGHT_ATTRS" : {"DEP" : "nummod"}},
             {"LEFT_ID" : "luna", "REL_OP" : ".", "RIGHT_ID" : "an", "RIGHT_ATTRS" : {"DEP" : "nummod"}}
            ],

            # data nasterii 20 mai 1989
            [{"RIGHT_ID" : "anch_data", "RIGHT_ATTRS" : {"LEMMA" : "dată"}},
             {"LEFT_ID" : "anch_data", "REL_OP" : ">", "RIGHT_ID" : "anch_nastere", "RIGHT_ATTRS" : {"ORTH" : {"IN" : nastere_syn}}},
             {"LEFT_ID" : "anch_data", "REL_OP" : ">", "RIGHT_ID" : "luna", "RIGHT_ATTRS" : {"DEP" : "nmod"}},
             {"LEFT_ID" : "luna", "REL_OP" : ";", "RIGHT_ID" : "zi", "RIGHT_ATTRS" : {"DEP" : "nummod"}},
             {"LEFT_ID" : "luna", "REL_OP" : ".", "RIGHT_ID" : "an", "RIGHT_ATTRS" : {"DEP" : "nummod"}}
            ],
        ]
        matcher.add("birthdate2", birthdate_pattern2)
    
        birthdate_pattern3 = [
    
            # nascuta la 20.05.1989
            [{"RIGHT_ID" : "anch_nascut", "RIGHT_ATTRS" : {"ORTH" : {"IN" : nascut_syn}}},
             {"LEFT_ID" : "anch_nascut", "REL_OP" : ">", "RIGHT_ID" : "data", "RIGHT_ATTRS" : {"DEP" : {"IN" : ["nummod", "nmod"]},
                                                                                             "SHAPE" : {"IN" : date_shape}}}
            ],
        ]
        matcher.add("birthdate3", birthdate_pattern3)  

        birthdate_pattern4 = [
    
            # nascuta pe 20 mai 1989
            [{"RIGHT_ID" : "anch_nascut", "RIGHT_ATTRS" : {"ORTH" : {"IN" : nascut_syn}}},
             {"LEFT_ID" : "anch_nascut", "REL_OP" : ">", "RIGHT_ID" : "luna", "RIGHT_ATTRS" : {"DEP" : {"IN" : ["nummod", "nmod", "obl"]}}},
             {"LEFT_ID" : "luna", "REL_OP" : ";", "RIGHT_ID" : "zi", "RIGHT_ATTRS" : {"DEP" : "nummod"}},
             {"LEFT_ID" : "luna", "REL_OP" : ".", "RIGHT_ID" : "an", "RIGHT_ATTRS" : {"DEP" : "nummod"}}
            ],
        ]
        matcher.add("birthdate4", birthdate_pattern4)   
        
        return matcher
    
    def get_birthdate_interval(self, doc, match):
        
        match_name = doc.vocab.strings[match[0]]
        match_tokens = match[1]
        
        if match_name == 'birthdate1':
            start = doc[match_tokens[2]].idx
            end = start + len(doc[match_tokens[2]].text) 
            
        elif match_name == 'birthdate2':
            start = doc[match_tokens[3]].idx
            end = doc[match_tokens[4]].idx + len(doc[match_tokens[4]].text)
            
        elif match_name == 'birthdate3':
            start = doc[match_tokens[1]].idx
            end = start + len(doc[match_tokens[1]].text) 
        
        elif match_name == 'birthdate4':
            start = doc[match_tokens[2]].idx
            end = doc[match_tokens[3]].idx + len(doc[match_tokens[3]].text)
            
        return start, end
    
    def match_birthdate(self, doc, text):
        """ Return the position of all the matches for birthdate in a text. """
        
        res = {}
        
        matcher = self.birthdate_matcher()
        spacy_matches = matcher(doc)
        
        for match in spacy_matches:        
            start, end = self.get_birthdate_interval(doc, match)
            res[start] = [start, end, 'NASTERE']
            
            if self.debug: 
                print(match)
                
        return res
    
    def match_cui(self, text):
        """ Return the position of all the matches for CUIs and Js in a text. """
        
        matches = re.findall(ALL_CUI_REGS, text)
        
        res = {}
        for match in matches: 
            start, end = self.find_match(match, text, res)
            res[start] = [start, end, 'CUI']
            
            if self.debug: 
                print(match)
                
        return res
    
    def check_common_words(self, matches, doc):
        """ Split matches in sequence of words which do not contain common words. """
        
        new_matches = []
        
        for (start, end) in matches:
    
            current_match_start = -1
            current_match_end = -1
    
            for i in range(start, end):
                token = doc[i]
    
                if not token.is_alpha or token.text[0].isupper():
                    if current_match_start == -1:
                        current_match_start = i
                    current_match_end = i + 1
    
                if (token.is_alpha and not token.text[0].isupper()) or i == end - 1:
                    if current_match_start != -1:
    
                        new_matches.append((current_match_start,current_match_end))
    
                        current_match_end = -1
                        current_match_start = -1
                    
        return new_matches
    
    def match_brand(self, nlp, text,
                    brand_checks=[]
                   ):
        """ Return the position of all the matches for brands in a text. """
        
        if type(brand_checks) == int:
            brand_checks = [brand_checks]
        
        doc = nlp(text)    
        
        # Form the list of candidate matches
        candidate_matches = []
        for ent in doc.ents:
            if ent.label_ == 'PRODUCT' or (BRAND_INCLUDE_FACILITY in brand_checks and ent.label_ == 'FACILITY'):
                candidate_matches.append((ent.start, ent.end))
        
        # Exclude common words
        if BRAND_EXCLUDE_COMMON in brand_checks:
            candidate_matches = self.check_common_words(candidate_matches, doc)     
        
        # Get final matches
        final_matches = {}
        for (start, end) in candidate_matches:
            start_idx = doc[start].idx
            end_idx = doc[end - 1].idx + len(doc[end - 1])
            final_matches[start_idx] = [start_idx, end_idx, "BRAND"]
            
            if self.debug: 
                print(doc[start:end])
                
        return final_matches
    
    def match_registry(self, text):
        """ Return the position of all the matches for Registry in a text. """
        
        # Build Registry REGEX
        registry_regex = r'(' + self.registry_keywords + ')' + REGISTRY_REG + REG_END
       
        matches = re.findall(registry_regex, text)
            
        res = {}
        for groups in matches:
            match = groups[2]
            
            start, end = self.find_match(match, text, res)
            res[start] = [start, end, 'SERIE']
                
            if self.debug: 
                print(match)
                
        return res
    
    def match_eu_case(self, text):
        """ Return the position of all the matches for EU cases in a text. """
        
        matches = re.findall(ALL_EU_CASE_REGS, text)
        
        res = []
        for match in matches:   
            res.append(self.find_match(match, text, res))
            
            if self.debug: 
                print(match)
                
        return res
    
    def ignore_near_case_matches(self, matches, case_matches):
        """ Remove matches close to an EU case match """
        
        new_matches = {}
        for (m_start, m_end, m_type) in matches.values():
            
            near_match = False        
            for (c_start, c_end) in case_matches:
                
                if max(c_start, m_start) - min(c_end, m_end) < MIN_CASE_DISTANCE:
                    near_match = True
                    break
                    
            if not near_match:
                new_matches[m_start] = (m_start, m_end, m_type)
                
            elif self.debug: 
                print('Removed', (m_start, m_end, m_type))
                
        return new_matches
        
    #######
    # AUX #
    #######
    
    
    
    
    def _pre_process(self, inputs):
                
        text = inputs['DOCUMENT']
        if len(text) < ct.MODELS.TAG_MIN_INPUT:
          raise ValueError("Document: '{}' is below the minimum of {} words".format(
            text, ct.MODELS.TAG_MIN_INPUT))
          
        self.debug = bool(inputs.get('DEBUG', False))
        
        # TODO Remove
        # Normalize institution names
        self.new_institution_list = []
        for inst in self.institution_list:
            inst_normalized = unidecode.unidecode(inst.lower())
            self.new_institution_list.append(inst_normalized)
            
            inst_stripped = ''.join(filter(str.isalnum, inst_normalized))
            self.new_institution_list.append(inst_stripped)
        
        # Apply spaCy analysis
        doc = self.nlp_model(text)
    
        return text, doc

    def _predict(self, prep_inputs):
        
        text, doc = prep_inputs    
        
        matches = {}
        
        # Match CNPS
        matches.update(self.match_cnp(text))
        
        # Match Serie Numar CI
        matches.update(self.match_serie_numar(text))
    
        # Match email
        matches.update(self.match_email(text))
        
        # Match names
        name_matches, person_dict = self.match_name(self.nlp_model, doc, text, 
                                                    person_checks=[PERSON_PROPN, PERSON_UPPERCASE])
        
        matches.update(name_matches)  
        if self.debug:
            print(person_dict)
            
        # Match addresses
        matches.update(self.match_address(self.nlp_model, doc, text, 
                                          address_checks=[ADDRESS_INCLUDE_GPE, ADDRESS_REMOVE_PUNCT]))

        # Match phone
        matches.update(self.match_phone(text, check_strength=PHONE_REG_VALID))
        
        # Match IBAN
        matches.update(self.match_iban(text))
        
        # Match institutions
        matches.update(self.match_institution(self.nlp_model, text, public_insts=self.new_institution_list))
        
        # Match birthdate
        matches.update(self.match_birthdate(doc, text))
        
        # Match CUI
        matches.update(self.match_cui(text))
        
        # Match Brand
        matches.update(self.match_brand(self.nlp_model, text, 
                                        brand_checks=[BRAND_EXCLUDE_COMMON, BRAND_INCLUDE_FACILITY]))
                                        
        # Match registry
        matches.update(self.match_registry(text))
        
        # Match EU case and ignore nearby matches
        cases = self.match_eu_case(text)
        matches = self.ignore_near_case_matches(matches, cases)
        
              
        return text, matches, person_dict

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
                hidden_doc = hidden_doc[:start] + 'X' + hidden_doc[end:]
        
        # Replace names with their codes, starting with longer names (which might include the shorter ones)
        for name in sorted(person_dict, key=len, reverse=True):
            code = person_dict[name]
            
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
      'DEBUG' : True,
      
#       'DOCUMENT': """S-au luat în examinare ADPP apelurile declarate de Parchetul de pe lângă Înalta Curte de Casaţie şi Justiţie – Direcţia naţională Anticorupţie şi de inculpatul Popa Vasile Constantin împotriva sentinţei penale nr. 194/PI din 13 martie 2018 a Curţii de Apel Timişoara – Secţia Penală.
# Dezbaterile au fost consemnate în încheierea de şedinţă din data de 09 ianuarie 2020,  ce face parte integrantă din prezenta decizie şi având nevoie de timp pentru a delibera, în baza art. 391 din Codul de procedură penală a amânat pronunţarea pentru azi 22 ianuarie 2020, când în aceeaşi compunere a pronunţat următoarea decizie:
# ÎNALA CURTE
#  	Asupra apelurilor penale de faţă;
# În baza lucrărilor din dosar, constată următoarele:
# Prin sentinţa penală nr. 194/PI din 13 martie 2018 a Curţii de Apel Timişoara – Secţia Penală, pronunţată în dosarul nr.490/35/2014, în baza art. 386 din Codul de procedură penală a respins cererea de schimbare a încadrării juridice a faptei de sustragere sau distrugere de înscrisuri, prev. de art. 242 al. 1 şi 3 din Codul penal, cu aplic. art. 5 din Codul penal, în cea de sustragere sau distrugere de probe ori de înscrisuri, prev. de art. 275 al. 1 şi 2 din Codul penal, formulată de inculpatul POPA VASILE CONSTANTIN
# """,
      
       # 'DOCUMENT': """Se desemnează domnul Cocea Radu, avocat, cocea@gmail.com, 0216667896 domiciliat în municipiul Bucureşti, Bd. Laminorului nr. 84, sectorul 1, legitimat cu C.I. seria RD nr. 040958, eliberată la data de 16 septembrie 1998 de Secţia 5 Poliţie Bucureşti, CNP 1561119034963, în calitate de administrator special. Se desemneaza si doamna Alice Munteanu cu telefon 0216654343, domiciliata in Bd. Timisoara nr. 107 """, 
        
      # 'DOCUMENT': """Cod numeric personal: 1505952103022. Doi copii minori înregistraţi în documentul de identitate.""",
        
       # 'DOCUMENT': """Bătrîn Cantemhir-Marian, Str. Cardos Iacob nr. 34, Arad, judeţul Arad, 1850810020101. 
       # Almăjanu Steliana, Comuna Peretu, judeţul Teleorman, 2580925341708.""",
      
#       'DOCUMENT' : """
# III. În baza art. 396 al. 1 şi 5 din Codul de procedură penală rap. la art. 16 al. 1 lit. b din Codul de procedură penală a fost achitat inculpatul MIHALACHE GABRIEL-CONSTANTIN, fiul lui Marin şi Marioara - Aurora, născut la 18.05.1952 în Brad, jud. Hunedoara, domiciliat în Oradea, strada Episcop Ioan Suciu nr.4, bloc ZP2, apt.10, CNP 1520518054675, pentru săvârşirea infracţiunii de efectuarea unei prelevări atunci când prin aceasta se compromite o autopsie medico-legală, prev. de art. 155 din Legea nr. 95/2006 republicată.
#  	În baza art. 397 al. 1 din Codul de procedură penală s-a luat act că persoanele vătămate Lozincă Maria, Ministerul Public – Parchetul de pe lângă Înalta Curte de Casaţie şi Justiţie şi Parchetul de pe lângă Tribunalul Bihor nu 
# s-au constituit părţi civile în cauză.
#  	În baza art. 274 al. 1 din Codul de procedură penală a fost obligat inculpatul Popa Vasile Constantin cu serie RK897456 la plata sumei de 20.000 lei cu titlu de cheltuieli judiciare faţă de stat.
#  	În baza art. 275 al. 3 din Codul de procedură penală cheltuielile judiciare ocazionate cu soluţionarea cauzei faţă de inculpaţii David Florian Alin şi Mihalache Gabriel Constantin, au rămas în sarcina statului.
#  	În baza art. 275 al. 6 din Codul de procedură penală s-a dispus plata din fondurile Ministerului Justiţiei către Baroul Timiş a sumei de câte 350 lei, reprezentând onorariu parţial avocat din oficiu către avocaţii Schiriac Lăcrămioara şi Miloş Raluca, respectiv 100 lei, reprezentând onorariu parţial avocat din oficiu către avocatul Murgu Călin.
#  	În baza art. 120 al. 2 lit. a teza 2 din Codul de procedură penală a fost respinsă cererea de acordare a cheltuielilor de transport pentru termenul din 2 noiembrie 2016, formulată de martora Bodin Alina Adriana.
#  	În baza art. 120 al. 2 lit. a teza 2 din Codul de procedură penală a fost admisă în parte cererea formulată de martorul Popa Vasile cu privire la acordarea cheltuielilor legate de deplasarea la instanţă. 
# S-a dispus plata din fondurile Ministerului Justiţiei a sumei de 480 lei, reprezentând contravaloarea serviciilor de cazare privind pe martorul Popa Vasile, pentru termenele de judecată din 31 octombrie 2017 şi 7 noiembrie 2017 şi respinge în rest cererea formulată.
# Pentru a pronunţa această sentinţă, prima instanţă a reţinut că prin rechizitoriul nr. 421/P/2013 din 17.12.2014 al Parchetului de pe lângă Înalta Curte de Casaţie şi Justiţie – Direcţia Naţională Anticorupţie, înregistrat la Curtea de Apel Oradea sub nr. dosar 490/35/2014 la data de 19.12.2014, s-a dispus trimiterea în judecată a inculpaţilor POPA VASILE CONSTANTIN, pentru săvârşirea infracţiunii de luare de mită, prev. de art. 6 din L. 78/2000 rap. la art. 289 din Codul penal  rap. la art. 7 al. 1 lit. b din L. 78/2000; infracţiunii de şantaj, prev. de art. 207 al. 1 din Codul penal, cu aplic. art. 13 ind. 1 din L. 78/2000; 3 infracţiuni de abuz în serviciu, prev. de art. 13 ind. 2 din L. 78/2000 cu referire la art. 297 al. 1 din Codul penal; 3 infracţiuni de distrugere de înscrisuri prev. de art. 242 al. 1 şi 3 din Codul penal; 3 infracţiuni de favorizare a făptuitorului, prev. de art. 269 al. 1 din Codul penal; infracţiunii de instigare la efectuarea unor prelevări atunci când prin aceasta se compromite o autopsie medico-legală, prev. de art. 47 din Codul penal rap. la art. 156 din Legea nr. 95/2006 şi a infracţiunii de trafic de influenţă prev. de art. 6 din Legea nr. 78/2000 rap. la art. 291 din Codul penal rap. la art. 7 al. 1 lit. b din Legea nr. 78/2000; MIHALACHE GABRIEL CONSTANTIN, pentru săvârşirea infracţiunii de efectuare a unei prelevări atunci când prin aceasta se compromite o autopsie medico-legală, prev. de art. 156 din Legea nr. 95/2006 republicată şi DAVID FLORIAN ALIN, pentru săvârşirea infracţiunii de cumpărare de influenţă, prev. de art. 6 din Legea nr. 78/2000 rap. la art. 292 din Codul penal, cu aplicarea art. 5 din Codul penal. 

# """
    # 'DOCUMENT' : """Subsemnatul Damian Ionut Andrei, domiciliat in Voluntari, str. Drumul Potcoavei nr 120, bl B, 
    # sc B, et 1, ap 5B, avand CI cu CNP 1760126413223, declar pe propria raspundere ca sotia mea Andreea Damian, 
    # avand domiciliul flotant in Cluj, Strada Cernauti, nr. 17-21, bl. J, parter, ap. 1 nu detine averi ilicite""",
    
    # 'DOCUMENT' : """Subsemnatul Damian Ionut Andrei, domiciliat in Cluj, Strada Cernauti, nr. 17-21, bl. J, parter, ap. 1 , nascut pe data 24-01-1982, declar pe propria raspundere ca sotia mea Andreea Damian, avand domiciliul flotant in Bucuresti, str. Drumul Potcoavei nr 120, bl. B, sc. B, et. 1, ap 5B, avand CI cu CNP 1760126413223 serie RK, numar 897567 nu detine averi ilicite""",
    
    # 'DOCUMENT' : """decizia recurată a fost dată cu încălcarea autorităţii de lucru interpretat, respectiv cu încălcarea dispozitivului hotărârii preliminare pronunţate de Curtea de Justiţie a Uniunii Europene în Cauza C-52/07 (hotărâre care are autoritate de lucru interpretat „erga omnes”)""",
    
    # 'DOCUMENT' : """Subsemnatul Laurentiu Piciu, data nastere 23.07.1995, loc nastere in Rm. Valcea, jud. Valcea, Bd. Tineretului 3A, bl A13, angajat al 
    # S.C. Knowledge Investment Group S.R.L. CUI 278973, cu adresa in Sector 3 Bucuresti, Str. Frunzei 26 et 1, va rog a-mi aproba cererea de concediu pentru 
    # perioada 16.02.2022 - 18.02.2022"""
    
    # 'DOCUMENT' : """Majorează de la 100 lei lunar la câte 175 lei lunar contribuţia de întreţinere datorată de pârâtă reclamantului, în favoarea minorilor A... C... R... Cezărel nascut la data de 20.02.2001 şi A... D... D... născută la data de 07 iunie 2002, începând cu data"""
    
    # DE LA CLIENT
    
    # 'DOCUMENT' : """Ciortea Dorin, fiul lui Dumitru şi Alexandra, născut la 20.07.1972 în Dr.Tr.Severin, jud. Mehedinţi, domiciliat în Turnu Severin, B-dul Mihai Viteazul nr. 6, bl.TV1, sc.3, et.4, apt.14, jud. Mehedinţi, CNP1720720250523, din infracțiunea prevăzută de art. 213 alin.1, 2 şi 4 Cod penal în infracțiunea prevăzută de art. 213 alin. 1 şi 4 cu aplicarea art.35 alin. 1 Cod penal (persoane vătămate Zorliu Alexandra Claudia şi Jianu Ana Maria).""",
    
    # 'DOCUMENT' : """II. Eşalonul secund al grupului infracţional organizat este reprezentat de inculpaţii Ruse Adrian, Fotache Victor, Botev Adrian, Costea Sorina şi Cristescu Dorel.""",
    
    # 'DOCUMENT' : """Prin decizia penală nr.208 din 02 noiembrie 2020 pronunţată în dosarul nr. 2187/1/2020 al Înaltei Curţi de Casaţie şi Justiţie, Completul de 5 Judecători a fost respins, ca inadmisibil, apelul formulat de petentul Dumitrescu Iulian împotriva deciziei penale nr.111 din 06 iulie 2020 pronunţată în dosarul nr. 1264/1/2020 al Înaltei Curţi de Casaţie şi Justiţie, Completul de 5 Judecători.""",
    
    # 'DOCUMENT' : """În momentul revânzării imobilului BIG Olteniţa către Ruse Adrian pe SC Casa Andreea , preţul trecut în contract a fost de 1.500.000 lei, însă preţul a fost fictiv, acesta nu a fost predat în fapt lui Ruse Adrian.""",
    
#     'DOCUMENT' : """intimatul Ionescu Lucian Florin (fiul lui Eugen şi Anicuţa, născut la data de 30.09.1984 în municipiul Bucureşti, domiciliat în municipiul Bucureşti, str. Petre Păun, nr. 3 bl. G9D, sc. 3, et. 8, ap. 126, sector 5, CNP 1840930420032, prin sentinţa penală nr. 169 din 29.10.2015 a Tribunalului Călăraşi, definitivă prin decizia penală nr. 1460/A din 07.10.2016 a Curţii de Apel Bucureşti - Secţia a II-a Penală sunt concurente cu cele pentru a căror săvârşire a fost condamnat acelaşi intimat prin sentinţa penală nr. 106/F din 09.06.2016 pronunţată de Ionescu Florin.
# în mod corect şi motivat, Ionescu Lucian a fost declarat
# în mod corect şi motivat, lonescu Lucian Florin  a fost declarat""",

    # 'DOCUMENT' : """-a dispus restituirea către partea civilă Barbu Georgiana a tabletelor marca Sony Vaio cu seria SVD112A1SM, cu încărcător aferent şi marca ASUS seria SN:CCOKBC314490""",
    
    # 'DOCUMENT' : """Comanda comerciala nr. 1320679561/27 august 2014 efectuată de reprezentantul SC Sady Com SRL de la societatea Borealis L.A.T., prin care prima societate a achiziţionat o cantitate de îngrăşăminte la preţul de 44.684,64 lei,""",
    
    # 'DOCUMENT' : """Contractul comercial nr. 23/14 februarie 2014 încheiat între SC Sady Com SRL şi SC Managro SRL, prin care prima societate a vândut celei de-a doua cantitatea de 66 tone azotat de amoniu la preţul de 93.720 RON, precum şi factum proforma emisă de reprezentantul SC Sady Com SRL pentru suma de 93.720 RON.""",
    
    'DOCUMENT' : """În temeiul art. 112 alin. 1 lit. b) s-a dispus confiscarea telefonului marca Samsung model G850F, cu IMEI 357466060636794 si a cartelei SIM seria 8940011610660227721, folosit de inculpat în cursul activităţii infracţionale.""",
    
    # 'DOCUMENT' : """Relevant în cauză este procesul-verbal de predare-primire posesie autovehicul cu nr. 130DT/11.10.2018, încheiat între Partidul Social Democrat (în calitate de predator) și Drăghici Georgiana (în calitate de primitor) din care rezultă că la dată de 08 octombrie 2018 s-a procedat la predarea fizică către Drăghici Georgiana a autoturismului Mercedes Benz P.K.W model GLE 350 Coupe, D4MAT, serie șasiu WDC 2923241A047452, serie motor 64282641859167AN 2016 Euro 6, stare funcționare second hand – bună, precum și a ambelor chei. La rubrica observații, Partidul Social Democrat, prin Serviciul Contabilitate a constatat plata, la data de 08 octombrie 2018, a ultimei tranșe a contravalorii autovehiculului a dat catre Georgiana Drăghici."""
    
      }
  
  res = eng.execute(inputs=test, counter=1)
  print(res)
