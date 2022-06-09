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
from fuzzywuzzy import fuzz


_CONFIG = {
  'SPACY_MODEL' : 'ro_core_news_lg',
 }


# File paths
# Debug
INSTITUTION_LIST_DEBUG = 'C:\\Proiecte\\LegeAI\\Date\\nomenclator institutii publice.txt'
ORGANIZATION_LIST_DEBUG = 'C:\\Proiecte\\LegeAI\\Date\\Task6\\organizatii.txt'
CONF_REGEX_DEBUG = 'C:\\Proiecte\\LegeAI\\Date\\Task6\\conf_regex.json'
PREFIX_INSTITUTION_DEBUG = 'C:\\Proiecte\\LegeAI\\Date\\Task6\\prefix_institutii.txt'
# Prod
INSTITUTION_LIST_PROD = 'C:\\allan_data\\2022.01.26\\nomenclator institutii publice.txt'
ORGANIZATION_LIST_PROD = 'C:\\allan_data\\2022.01.26\\organizatii.txt'
CONF_REGEX_PROD = 'C:\\allan_data\\2022.03.07\\conf_regex.json'
PREFIX_INSTITUTION_PROD = 'C:\\allan_data\\2022.01.26\\prefix_institutii.txt'


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
SAME_NAME_THRESHOLD = 75
CHECK_SON_OF_INTERVAL = 40
SON_OF_PHRASES = ["fiul lui", "fiica lui"]
NEE_PHRASES = ['fostă', 'fosta', 'fost', 'nascuta', 'nascut', 'născut']

# PUBLIC INSTITUTIONS
MIN_PREFIX_FUZZY = 90

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
SERIE_FORMS = 'serie|SERIE|Serie|seria|SERIA|Seria' 
NUMAR_FORMS = 'numar|NUMAR|Numar|număr|NUMĂR|Număr|nr|NR|Nr'
SERIE_NUMAR_REGS = {
    r'(?:' + SERIE_FORMS + ') [A-Z]{2}.{0,5}(?:' + NUMAR_FORMS + ')(?:.)? \d{6}',
    r'(?:' + SERIE_FORMS + ') [A-Z]{2}\d{6}',
    r'(?:' + NUMAR_FORMS + ')(?:.)? [A-Z]{2}\d{6}'     
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
IBAN_REG = r'(?:RO|ro|Ro)[ ]?\d{2}[ ]?\w{4}(?:[ ]?[A-Z0-9]{4}){4}'
IBAN_REG = REG_START + IBAN_REG + REG_END

# CUI
CUI_REGS = {
    # CUI 278973
    r'(?:CUI|CIF|cui|cif|Cui|Cif)(?: )?(?:RO)?\d{2,10}',
    
    # RO278973
    r'(?:RO|ro|Ro)\d{2,10}',    
    
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

# ABBREVIATIONS
ALIN_REG =  r'(?<=\balin)( |si|sau|și|şi|\d|,|\.|_|-|\))*\d\b'
# Positive lookaround for word boundary but negative lookaround for paranthesis
ALIN_NUMBER_REG = r'\b\d+(?=\b)(?!\))'

# Overlap
NO_OVERLAP = 0
INTERSECTION = 1
INCLUDED_1IN2 = 2
INCLUDED_2IN1 = 3

# Punctuation
DOUBLE_PUNCTUATION = r'\({2}|,{2}|\/{2}|"{2}|:{2}|;{2}|\){2}|\\{2}|(?<!\.)\.{2}(?!\.)'
TOO_MANY_DOTS = r'\.{4,}'
SOLO_PUNCTUATION = r'(\.{3})|[\.,\:;\?\!]'
PAIR_PUNCTUATION = r'\(.+\)|\[.+\]|\{.+\}|\".+\"|\'.+\''

SPACY_LABELS = ['NUME', 'ADRESA', 'INSTITUTIE', 'NASTERE', 'BRAND']


__VER__='0.7.1.1'
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
                    print('CNP:', text[start:end])
                
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
                    print('Serie Numar CI:', text[start:end])
                        
            
        return res
    
    def next_name_code(self, code):
        """ Get the next code for a new name. """        
        
        if code[0] == 'Z':
            next_code = 'A' * (len(code) + 1)
        else:
            next_code = [chr(ord(c) + 1) for c in code]
            next_code = ''.join(next_code)
            
        return next_code
    
    def check_son_of_phrases(self, start, text):
        """ Check if phrases related to the Son Of construct appear in the prefix. """
        
        prefix = text[max(0, start - CHECK_SON_OF_INTERVAL) : start].lower()
        
        for phrase in SON_OF_PHRASES:
            if phrase.lower() in prefix:
                return True
            
        return False
    
    def unite_same_names(self, text):
        """ Search the name list for names that refer to the same entitiy and unite them. """
    
        different_ents = list(range(len(self.name_list)))
        
        for i in range(len(self.name_list)):
            start1 = self.name_list[i][0]
            name1 = self.name_list[i][1]
            
            for j in range(i+1, len(self.name_list)):
                start2 = self.name_list[j][0]
                name2 = self.name_list[j][1]
                            
                # Get the similarity between the names
                ratio = fuzz.token_set_ratio(name1, name2)            
                
                if ratio > SAME_NAME_THRESHOLD:
                    
                    if not self.check_son_of_phrases(start1, text) and not self.check_son_of_phrases(start2, text):
    
                        # Get the two corresponding entities
                        ent1 = different_ents[i]
                        ent2 = different_ents[j]
    
                        # The common new entity will be the minimum between them
                        min_ent = min(ent1, ent2)
    
                        # Set all occurances of the two entities to the minimum entity between them
                        for k in range(len(different_ents)):
                            if different_ents[k] == ent1 or different_ents[k] == ent2:
                                different_ents[k] = min_ent
        
        return different_ents
    
    def set_name_codes(self, text):
        """ Form the dictionary of names and codes. """
        
        # Sort the names according to their first position in the text
        self.name_list.sort(key=lambda tup: tup[0])
    
        # Unite names that refer to the same entity
        different_ents = self.unite_same_names(text) 
        
        codes = [''] * len(different_ents)        
        current_code = 'A'
    
        # Get a code for each of the entities
        for i, ent in enumerate(different_ents):
            # If no code was alreay set
            if codes[i] == '':
                
                # Set the current code for all occurances of the entity
                for j in range(i, len(different_ents)):
                    if different_ents[j] == ent:
                        codes[j] = current_code
                        
                # Get the next code
                current_code = self.next_name_code(current_code)
        
        # Set the name - code dictionary
        name_code_dict = {self.name_list[i][1] : codes[i] for i in range(len(self.name_list))}
        
        return name_code_dict
        
    
    def find_name(self, name, match='levenshtein'):
        """ Find a name in the list of names. """
        
        low_name = name.lower()
        
        for i, (pos, name) in enumerate(self.name_list):
            if match == 'levenshtein':
                lev_dist = simple_levenshtein_distance(name.lower(), low_name, 
                                                   normalize=False)
            elif match == 'perfect':
                lev_dist = MAX_LEV_DIST
                
            if name.lower() == low_name or lev_dist < MAX_LEV_DIST:
                return i
            
        return -1
    

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
    
    def clean_match(self, doc, start, end):
        """ Check and clean a match. """
        
        # Remove unwanted punctuation at the end
        if doc[end - 1].text in ['(', '-']:
            end -= 1
        # Remove unwanted punctuation at the start
        if doc[start].text in [')', '-']:
            start += 1
        
        # Check paranthesis to be matched
        left_par, right_par = False, False
        for i in range(start, end):
            token = doc[i]        
            if token.text == '(':
                left_par = True            
            if token.text == ')':
                if left_par == False:
                    # Unmatched right paranthesis, ignore added words
                    return -1, -1
                else:
                    right_par = True                
        if left_par == True and right_par == False:
            # Unmatched right paranthesis, ignore added words
            return -1, -1        
        
        return start, end 
    
    def match_name(self, nlp, doc, text,
                   person_checks=[]
                  ):
        """ Return the position of names in a text. """  
        
        if type(person_checks) == int:
            person_checks = [person_checks]
        
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
        match_list = []
        for (start, end) in candidate_matches:
            
            # Check words to the right    
            new_end = end
            while new_end < len(doc):
                if (doc[new_end].text[0].isupper() or 
                    doc[new_end].text in ['(', ')', '-'] or 
                    # If next word is a Nee phrase and the following word is a potential name (uppercase first letter)
                    (new_end + 1 < len(doc) and doc[new_end].text in NEE_PHRASES and doc[new_end + 1].text[0].isupper())):
                    # Add capitalized words to the right
                    new_end += 1
                else:
                    break
                
            # Check words to the left
            new_start = start
            while new_start > 0:
                if (doc[new_start - 1].text[0].isupper() or 
                    doc[new_start - 1].text in ['(', ')', '-'] or
                    doc[new_start - 1].text in NEE_PHRASES) and not doc[new_start - 1].is_sent_start:
                    # Add capitalized words to the left, except if they are at the start of a sentence 
                    new_start -= 1               
                else:
                    break
                
           
            # Check and cleand the new match
            new_start, new_end = self.clean_match(doc, new_start, new_end)
            if new_start == -1 and new_end == -1:
                new_start, new_end = start, end
            
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
                
                # Ignore leading and trailing punctuation (apart for paranthesis)
                while text[start_idx] in punctuation.replace('()', ''):
                    start_idx += 1
                while text[end_idx - 1] in punctuation.replace('()', ''):
                    end_idx -= 1
                    
                # Check if similar to previous match
                add_match = True
                if len(match_list):
                    prev_start_idx, prev_end_idx = match_list[-1]
                    if prev_start_idx <= start_idx and end_idx <= prev_end_idx:
                        add_match = False
                    elif start_idx <= prev_start_idx and prev_end_idx <= end_idx:
                        match_list[-1] = (start_idx, end_idx)
                        add_match = False
                
                if add_match:
                    match_list.append((start_idx, end_idx))    
                    
        # Form match dictionary
        match_dict = {}
        for (start, end) in match_list:
            match_dict[start] = (start, end, "NUME")
                
            person = text[start:end]
            if self.debug:
                print('Nume:', person)
                
            if self.find_name(person, match='perfect') == -1:
                self.name_list.append((start, person))
                
        return match_dict      
    
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
                            print('Adresa:', ent)
                        
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
                        print('Place:', ent)
    
        # Check all LOC entities from initial Doc
        matches = self.check_loc_entities(doc, matches, address_checks)  
        
        # Remove punctuation and check LOC entities again
        if ADDRESS_REMOVE_PUNCT in address_checks:
            
            # Get the new text, with punctuation removed
            new_text, token_pos_dict = self.remove_punct_tokens(doc)
    
            # Build a new spaCy Doc
            doc_rp = nlp(new_text)
            
            matches.update(self.check_loc_entities(doc_rp, matches, address_checks,
                                                   token_pos_dict=token_pos_dict))
                
        return matches
    
    def match_email(self, text):
        """ Return the position of all the matches for email in a text. """
        
        matches = re.findall(EMAIL_REG, text)
        
        res = {}
        for match in matches:  
            start, end = self.find_match(match, text, res)
            res[start] = [start, end, 'EMAIL']
            if self.debug:
                print('EMAIL:', text[start:end])
                
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
                        print('Telefon:', text[start:end])
                
        elif check_strength == PHONE_VALIDATION:
            matches = phonenumbers.PhoneNumberMatcher(text, "RO")
    
            for match in matches:
                start, end = self.find_match(match, text, res)
                res[start] = [start, end, 'TELEFON']
                if self.debug:
                    print('Telefon:', text[start:end])
            
        return res
    
    def text_to_lemmas(self, text, nlp):
        """ Transform the words in a text to the equivalent lemmas. """
        
        nlp_lower = nlp(text.lower())
        
        lemmas = []
        
        for token in nlp_lower:
            start_pos = token.idx
            
            lemma = token.lemma_        
            if text[start_pos].isupper():
                # Check if the original is capitalized
                lemma = lemma.capitalize()
                
            lemmas.append(lemma)
            
        return ' '.join(lemmas)
    
    def match_noconf_institution(self, text, public_insts):
        """ Match public institutions which should not be confidential. """
        
        text_lower = unidecode.unidecode(text.lower())
        
        matches = []
        for i, inst in enumerate(public_insts):
                
            for match in re.finditer("\\b" + re.escape(inst) + "\\b", text_lower):
                match_span = match.span()
                
                # Also remove Organization from name list (so a code is not generated)
                for i, (_, name) in enumerate(self.name_list):
                    name_lower = unidecode.unidecode(name.lower())
                    # Check if any name in name list includes the Organization
                    if inst in name_lower:
                        del self.name_list[i]
                        break
                
                add_match = True
                for prev_match in matches:
                    if prev_match == match_span:
                        add_match = False                            
                        break
                        
                if add_match:
                    matches.append(match_span)
                
        return matches
        
    def match_organization(self, nlp, text):
        """ Return the position of all the matches for organizations in a text. """
        
        matches = {}    
        
        doc = nlp(text)
        
        # Collect all Organization matches
        for ent in doc.ents:
            
            if ent.label_ == 'ORGANIZATION':
                matches[ent.start_char] = (ent.start_char, ent.end_char, "INSTITUTIE")
                if self.debug:
                    print('Organizatie spaCy:', ent.text)
                    
        # Search for matches in the organization file
        for org in self.organization_list:
            
            # Check all the matches
            for m in re.finditer(re.escape(org), text):
                matches[m.start()] = (m.start(), m.end(), "INSTITUTIE")
                
                if self.debug:
                    print('Organizatie txt:', org)                    
                
        # Add matches to list of names
        for (start, end, _) in matches.values():    
            organization = text[start:end]
            if self.find_name(organization) == -1:
                self.name_list.append((start, organization))
                
        return matches
    
    def match_iban(self, text):
        """ Return the position of all the matches for IBANs in a text. """
       
        matches = re.findall(IBAN_REG, text)
            
        res = {}
        for match in matches:
                
            start, end = self.find_match(match, text, res)
            res[start] = [start, end, 'CNP']
               
            if self.debug: 
                print('IBAN:', text[start:end])
        
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
                print('Data nastere:', text[start:end])
                
        return res
    
    def match_cui(self, text):
        """ Return the position of all the matches for CUIs and Js in a text. """
        
        matches = re.findall(ALL_CUI_REGS, text)
        
        res = {}
        for match in matches: 
            start, end = self.find_match(match, text, res)
            res[start] = [start, end, 'CUI']
            
            if self.debug: 
                print('CUI:', text[start:end])
                
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
                print('Brand:', doc[start:end])
                
        return final_matches
    
    def match_trigger_regex(self, text, trigger, regex):
        """ Match a Regex starting with a trigger """
        
        pos = text.find(trigger)
        
        matches = []
        
        while pos > -1:
            subtext = text[pos + len(trigger):]
            match = re.match(regex, subtext)
            
            if match:
                # Check all groups
                delta_pos = 0
                
                for group in match.groups():
                    # Find first occurance of group
                    group_pos = subtext.find(group)
                    
                    # Get actual position of group
                    start = pos + len(trigger) + delta_pos + group_pos
                    end = start + len(group)
                    add_match = True
                    
                    # Check if match should be concatenated with previous one
                    if matches:
                        (prev_start, prev_end) = matches[-1]
                        if prev_start <= start and start <= prev_end:
                            # Concatenate matches
                            matches[-1] = (min(prev_start, start), max(prev_end,end))
                            add_match = False
                            
                    if add_match:
                        # Add new match
                        matches.append((start, end))
                    
                    # Move forward in text
                    subtext = subtext[group_pos + 1:]
                    delta_pos += group_pos + 1
                    
            pos = text.find(trigger, pos + 1)
        
        # Form match dictionary
        match_dict = {}
        for (start, end) in matches:
            match_dict[start] = (start, end, trigger.upper())
            
        return match_dict
    
    def match_regex(self, text):
        """ Return the position of all the matches for a list of user defined REGEX. """
            
        res = {}
            
        for entry in self.conf_regex_list:
            regex = entry[1]
            
            if type(entry) is list:
                # Trigger + Regex
                
                if type(entry[0]) is list:
                    # Trigger list
                    trigger_list = []
                    for trigger in entry[0]:
                        # Add all variants of each trigger
                        trigger_list.extend([trigger.lower(), trigger.upper(), trigger.lower().title()])             
                else:
                    # Single trigger, add all variants                
                    trigger_list = [entry[0].lower(), entry[0].upper(), entry[0].lower().title()]
                
                for trigger in trigger_list:
                    res.update(self.match_trigger_regex(text, trigger, regex))
                    
            else:
                # Simple REGEX
                regex = entry
                tag = "REGEX"
                
                matches = re.findall(regex, text) 
                for match in matches:  
                    if type(match) is tuple:
                        # If there were multiple capturing groups, concatenate them
                        match = ''.join(match).strip()
    
                    start, end = self.find_match(match, text, res)
                    res[start] = [start, end, tag]
                    
        return res
    
    def match_eu_case(self, text):
        """ Return the position of all the matches for EU cases in a text. """
        
        matches = re.findall(ALL_EU_CASE_REGS, text)
        
        res = []
        for match in matches:   
            res.append(self.find_match(match, text, res))
            
            if self.debug: 
                print('EU Case', match)
                
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
    
    def filter_noconf_matches(self, text, matches, noconf_matches):
        """ Eliminate the matches which intersect with non confidential matches. """
        
        filtered_matches = []
        
        for match in matches:
            [start_match, end_match, label_match] = match
            
            add_match = True
            for noconf_match in noconf_matches:
                (start_filter, end_filter) = noconf_match
                
                if max(start_filter, start_match) < min(end_filter, end_match):
                    add_match = False
                    break
                    
            if add_match:
                filtered_matches.append(match)
            
            else:
                
                # If the match is in the name list, also remove it from there
                idx = self.find_name(text[start_match : end_match])
                if idx > -1:
                    del self.name_list[idx]
                
        return filtered_matches
    
    def filter_institution_matches_by_prefix(self, text, matches):
        """ Filter public institution (which should be confidential) from the matches using a list of prefixes. """
    
        filtered_matches = []
        noconf_matches = []
        
        for match in matches:
            filter_out = False
            
            if match[2] == 'INSTITUTIE':
                match_text = text[match[0] : match[1]]
                
                # Get the lemmas of the match
                match_lemmas = self.text_to_lemmas(match_text, self.nlp_model)
                
                for prefix in self.prefix_institution_list:
                    
                    # Fuzzy match between a prefix from the list and the prefix of the match
                    if fuzz.ratio(prefix, match_lemmas[:len(prefix)]) >= MIN_PREFIX_FUZZY:
                        
                        # Also remove the match from the name list  
                        idx = self.find_name(match_text)
                        if idx > -1:
                            del self.name_list[idx]
                           
                        filter_out = True
                        break
                        
            if not filter_out:
                filtered_matches.append(match)
            else:
                # Add the match as a public institution
                noconf_matches.append((match[0], match[1]))
        
        return filtered_matches, noconf_matches
    
    def select_matches(self, matches_dict, noconf_matches, text):
        """ Select the final matches. 
        1. Eliminate matches which are not confidential.
        2. Select matches which overlap. """
    
        matches = list(matches_dict.values())
        
        # 1. Eliminate matches which are not confidential
        matches = self.filter_noconf_matches(text, matches, noconf_matches)
        
        
        # 2. Eliminate public institutions which should not be confidential using a list of prefixes
        matches, new_noconf = self.filter_institution_matches_by_prefix(text, matches)
        noconf_matches.extend(new_noconf)
        
                
        # 3. Select matches which overlap
        
        # Sort matches ascending by start position
        sorted_matches = sorted(matches, key=lambda tup: tup[0])        
        final_matches = [sorted_matches[0]]
    
        # Check all matches
        for (start2, end2, label2) in sorted_matches[1:]:
    
            (start1, end1, label1) = final_matches[-1]
    
            # Check type of overlap
            if max(start1, start2) < min(end1, end2):
                # Intersection
                overlap = INTERSECTION           
                if start1 <= start2 and start2 < end1 and start1 < end2 and end2 <= end1:
                    # start1 < start2 < end2 < end1
                    overlap = INCLUDED_2IN1    
                elif start2 <= start1 and start1 < end2 and start2 < end1 and end1 <= end2:
                    # start2 < start1 < end1 < end2
                    overlap = INCLUDED_1IN2 
            else:
                # No overlap
                overlap = NO_OVERLAP          
    
            # If any kind of overlap
            if not overlap == NO_OVERLAP:
                print('overlap', label1, label2)
                
                # If only one is a spaCy match, give priority to non-spaCy match
                if label1 in SPACY_LABELS and label2 not in SPACY_LABELS:
                    final_matches[-1] = (start2, end2, label2)       
                    removed_match = text[start1:end1]
                
                elif label2 in SPACY_LABELS and label1 not in SPACY_LABELS:   
                    removed_match = text[start2:end2]      
                    
                # Otherwise check overlap type
                else:
                    if overlap == INTERSECTION:
                        start = min(start1, start2)
                        end = max(end1, end2)
            
                        label = label1 
                        removed_match = text[start1:end1]
                        if end2 - start2 > end1 - start1:
                            label = label2
                            removed_match = text[start2:end2]
            
                        final_matches[-1] = (start, end, label)
                        
                    elif overlap == INCLUDED_1IN2:
                        final_matches[-1] = (start2, end2, label2)   
                        removed_match = text[start1:end1]
                        
                    elif overlap == INCLUDED_2IN1: 
                        removed_match = text[start2:end2]   
                                 
                # If the match is in the name list, also remove it from there
                idx = self.find_name(removed_match)
                if idx > -1:
                    del self.name_list[idx]
                    
            else:                
                # No overlap
                final_matches.append((start2, end2, label2))
                
        # Form matches dictionary
        final_dict = {}
        for (start, end, label) in final_matches:
            final_dict[start] = (start, end, label)
    
        return final_dict, noconf_matches
    
    def replace_abbreviations(self, text):
        """ Use the abbreviations dictionary to set abbreviations. """
        
        replace_list = []
        
        for (key, value) in self.abbr_dict.items():
            
            if type(value) is list:
                abbr = value[0]
                conditions = value[1]            
            else:
                abbr = value
                conditions = []
                
            for match in re.finditer('\\b' + re.escape(key) + "\\b", text):
                start, end = match.span()
                
                if 'nostart' in conditions:
                    tokens = self.nlp_model.char_span(start, end)
                    if tokens and not tokens[0].is_sent_start:
                        replace_list.append((start, end, abbr))
                        
                else:
                    replace_list.append((start, end, abbr))                    
                
        # Sort replace list descending by start position
        replace_list.sort(key=lambda tup : tup[0], reverse=True)
                
        # Replace abbreviations
        for (start, end, abbr) in replace_list:
            text = text[:start] + abbr + text[end:]
                
        return text
    
    def replace_alin(self, text):
        """ Replace 'alin' in text """
            
        replace_list = []
        
        for match in re.finditer(ALIN_REG, text):
            alin_text = match.group()
            span_start, span_end = match.span()
            print('alin', alin_text)
            
            for number in re.finditer(ALIN_NUMBER_REG, alin_text):
                number_text = number.group()
                number_start, number_end = number.span()
                
                replace_list.append((span_start + number_start, span_start + number_end, number_text))
                
                    
        # Sort replace list descending by start position
        replace_list.sort(key=lambda tup : tup[0], reverse=True)
                
        # Add paranthesis to numbers
        for (start, end, number) in replace_list:
            text = text[:start] + number + ')' + text[end:]
                
        return text
    
    def clean_punctuation(self, text):
        """ Clean punctuation """
        
        # Remove double punctuation marks sequences
        re_matches = list(re.finditer(DOUBLE_PUNCTUATION, text))
        re_matches.reverse()
        for re_match in re_matches:
            pos = re_match.span()[0]
            # Remove initial dot
            text = text[:pos] + text[pos + 1:]
            
        # Reduce many dots to just 3
        re_matches = list(re.finditer(TOO_MANY_DOTS, text))
        re_matches.reverse()
        for re_match in re_matches:
            start, end = re_match.span()
            # Remove extra dots
            text = text[:start] + '...' + text[end:]
            
        # Spaces around solo punctuation
        re_matches = list(re.finditer(SOLO_PUNCTUATION, text))
        re_matches.reverse()
        for re_match in re_matches:
            start, end = re_match.span()
            char = text[start:end]
            
            # Skip spaces before
            while start > 0 and text[start - 1] == ' ':
                start -= 1
              
            if end < len(text) and text[end] == ' ':
                # Skip extra spaces after first space
                while end < len(text) - 1 and text[end + 1] == ' ':
                    end += 1       
            elif end < len(text) - 1 and text[end] != ' ':
                # Check if a space should be added after
                if (text[end - 1] == '.' and ((text[end].isdigit() and start > 0 and text[start - 1].isdigit()) or 
                                             # If before and after dot there is a digit
                                            (text[end].isupper() and start > 0 and text[start - 1].isupper()))):
                                            # If before and after dot there is a uppercase letter
                    pass
                    
                elif text[end].isalnum():
                    # Otherwise, an extra space is needed after
                    char += ' '
            
            # Remove extra spaces
            text = text[:start] + char + text[end:]
            
        # Spaces around pair punctuation
        re_matches = list(re.finditer(PAIR_PUNCTUATION, text))
        re_matches.reverse()
        for re_match in re_matches:
            start, end = re_match.span()
            seq = text[start:end]
            prefix = ''
            suffix = ''
                    
            # Check before first punctuation        
            if start > 0 and text[start - 1] == ' ':
                # Skip extra spaces before
                while start > 1 and text[start - 2] == ' ':
                    start -= 1
            elif start > 0 and text[start - 1] != ' ':
                # Add space before
                prefix = ' '
                
            # Check after first punctuation
            while len(seq) > 1 and seq[1] == ' ':
                seq = seq[:1] + seq[2:]
                
            # Check before second punctuation
            while len(seq) > 2 and seq[-2] == ' ':
                seq = seq[:-2] + seq[-1]
            
            # Check after second punctuation        
            if end < len(text) and text[end] == ' ':
                # Skip extra spaces after
                while end + 1 < len(text) and text[end + 1] == ' ':
                    end += 1
            elif end < len(text) and text[end] != ' ' and text[end].isalpha():
                # Add space after
                suffix = ' '      
            
            # Remove extra spaces
            text = text[:start] + prefix + seq + suffix + text[end:]    
          
        return text    
    
    
    
        
    #######
    # AUX #
    #######
    
    
    
    
    def _pre_process(self, inputs):
        
        self.debug = bool(inputs.get('DEBUG', False))
        
        # Read files
        if self.debug:
            institution_path = INSTITUTION_LIST_DEBUG
            organization_path = ORGANIZATION_LIST_DEBUG
            prefix_institution_path = PREFIX_INSTITUTION_DEBUG
            regex_path = CONF_REGEX_DEBUG
        else:
            institution_path = INSTITUTION_LIST_PROD
            organization_path = ORGANIZATION_LIST_PROD
            prefix_institution_path = PREFIX_INSTITUTION_PROD
            regex_path = CONF_REGEX_PROD
        
        # Read list of public institutions
        institution_file = open(institution_path, 'r', encoding='utf-8')
        self.institution_list = institution_file.read().splitlines()
        
        # Read list of prefixes for public institutions
        prefix_file = open(prefix_institution_path, 'r', encoding='utf-8')
        self.prefix_institution_list = prefix_file.read().splitlines()
        # Replace with lemmas
        self.prefix_institution_list = [self.text_to_lemmas(prefix, self.nlp_model) for prefix in self.prefix_institution_list]
    
        # Read list of organizations
        organization_file = open(organization_path, 'r', encoding='utf-8')
        self.organization_list = organization_file.read().splitlines()
        
        # Read JSON REGEX file
        json_file = open(regex_path, 'r', encoding="utf-8")
        json_string = json_file.read().replace('\\', '\\\\')
        json_data = json.loads(json_string)        
        self.conf_regex_list = json_data['conf_regex']
        self.abbr_dict = json_data['abbr_dict']                
        
        text = inputs['DOCUMENT']
        if len(text) < ct.MODELS.TAG_MIN_INPUT:
          raise ValueError("Document: '{}' is below the minimum of {} words".format(
            text, ct.MODELS.TAG_MIN_INPUT))
          
        # Remove extra whitespaces
        text = re.sub(' +', ' ', text)  
        
        # Clean punctuation
        text = self.clean_punctuation(text)        
        
        # TODO De sters cand institutiile vor fi deja normalizate la citire
        # Normalize institution names            
        self.new_institution_list = []
        for inst in self.institution_list:
            inst_normalized = unidecode.unidecode(inst.lower())
            self.new_institution_list.append(inst_normalized)
            
            inst_stripped = inst_normalized.replace('.', '')
            if inst_stripped != inst_normalized:
                self.new_institution_list.append(inst_stripped)
                
        # Replace with abbreviations
        text = self.replace_abbreviations(text)
        
        # Replace 'alin' sequences
        text = self.replace_alin(text)
        
        # Apply spaCy analysis
        doc = self.nlp_model(text)
        
        self.name_list = [] 
    
        return text, doc

    def _predict(self, prep_inputs):
        
        text, doc = prep_inputs    
        
        matches = {}
        
        # Match organizations
        matches.update(self.match_organization(self.nlp_model, text))
        
        # Match CNPS
        matches.update(self.match_cnp(text))
        
        # Match Serie Numar CI
        matches.update(self.match_serie_numar(text))
    
        # Match email
        matches.update(self.match_email(text))
        
        # Match names
        matches.update(self.match_name(self.nlp_model, doc, text, 
                                       person_checks=[PERSON_PROPN, PERSON_UPPERCASE]))       
                    
        # Match addresses
        matches.update(self.match_address(self.nlp_model, doc, text, 
                                          address_checks=[ADDRESS_INCLUDE_GPE, ADDRESS_REMOVE_PUNCT]))

        # Match phone
        matches.update(self.match_phone(text, check_strength=PHONE_REG_VALID))
        
        # Match IBAN
        matches.update(self.match_iban(text))
        
        # Match birthdate
        matches.update(self.match_birthdate(doc, text))
        
        # Match CUI
        matches.update(self.match_cui(text))
        
        # Match Brand
        matches.update(self.match_brand(self.nlp_model, text, 
                                        brand_checks=[BRAND_EXCLUDE_COMMON, BRAND_INCLUDE_FACILITY]))
                        
        # Match user REGEX
        matches.update(self.match_regex(text)) 
        
        # Match EU case and ignore nearby matches
        cases = self.match_eu_case(text)
        matches = self.ignore_near_case_matches(matches, cases)
        
        # Find entities which should not be confidential        
        noconf_matches = self.match_noconf_institution(text, public_insts=self.new_institution_list)
              
        return text, matches, noconf_matches

    def _post_process(self, pred):      
        
        text, matches, noconf_matches = pred
        
        # Select final matches
        matches, noconf_matches = self.select_matches(matches, noconf_matches, text)  
        
        # Order matches 
        match_tuples = list(matches.values())
        match_starts = list(sorted(matches.keys(), reverse=True))     
    
        # Replace all confidential information (except names) in text
        hidden_doc = text
        for key in match_starts:
            [start, end, label] = matches[key]
            if not label in ['NUME', 'INSTITUTIE']:
                hidden_doc = hidden_doc[:start] + 'x' + hidden_doc[end:]
                
        # Set the codes for the names
        name_code_dict = self.set_name_codes(text)  
            
        if self.debug:
            print(name_code_dict)   
        
        # Replace names with their codes, starting with longer names (which might include the shorter ones)
        for name in sorted(name_code_dict, key=len, reverse=True):
            code = name_code_dict[name]
            
            # Search for all occurances of name
            while True:
                # Ignore the case of the letters
                name_match = re.search(re.escape(name.lower()), hidden_doc.lower())
                
                if name_match:
                    start, end = name_match.span()
                    if end < len(hidden_doc) and hidden_doc[end] != '.':
                        # Add dot after the code if it's not before a dot
                        code += '.'
                    hidden_doc = hidden_doc[:start] + code + hidden_doc[end:]
                else:
                    break  
            
        # Add non confidential matches to result
        for (start, end) in noconf_matches:
            match_tuples.append((start, end, "ORGANIZATIE_PUBLICA"))
            
    
        # Clean punctuation again, after the matches have been replaces
        hidden_doc = self.clean_punctuation(hidden_doc) 
            
            
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
       
        # 'DOCUMENT': """Se desemnează domnul Cocea Radu, avocat, cocea@gmail.com, 0216667896 domiciliat în municipiul Bucureşti, Bd. Laminorului nr. 84, sectorul 1, legitimat cu C.I. Seria RD Nr. 040958, eliberată la data de 16 septembrie 1998 de Secţia 5 Poliţie Bucureşti, CNP 1561119034963, în calitate de administrator special. Se desemneaza si doamna Alice Munteanu cu telefon 0216654343, domiciliata in Bd. Timisoara nr. 107 """, 
        
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
    # S.C. Knowledge Investment Group S.R.L. Cui 278973, cu adresa in Sector 3 Bucuresti, Str. Frunzei 26 et 1, va rog a-mi aproba cererea de concediu pentru 
    # perioada 16.02.2022 - 18.02.2022"""
    
    # 'DOCUMENT' : """Majorează de la 100 lei lunar la câte 175 lei lunasr contribuţia de întreţinere datorată de pârâtă reclamantului, în favoarea minorilor A... C... R... Cezărel nascut la data de 20.02.2001 şi A... D... D... născută la data de 07 iunie 2002, începând cu data"""
    
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
    
    # 'DOCUMENT' : """În temeiul art. 112 alin. 1 lit. b) s-a dispus confiscarea telefonului marca Samsung model G850F, cu IMEI 357466060636794 si a cartelei SIM seria 8940011610660227721, folosit de inculpat în cursul activităţii infracţionale.""",
    
    # 'DOCUMENT' : """Relevant în cauză este procesul-verbal de predare-primire posesie autovehicul cu nr. 130DT/11.10.2018, încheiat între Partidul Social Democrat (în calitate de predator) și Drăghici Georgiana (în calitate de primitor) din care rezultă că la dată de 08 octombrie 2018 s-a procedat la predarea fizică către Drăghici Georgiana a autoturismului Mercedes Benz P.K.W model GLE 350 Coupe, D4MAT, serie șasiu WDC 2923241A047452, serie motor 64282641859167AN 2016 Euro 6, stare funcționare second hand – bună, precum și a ambelor chei. La rubrica observații, Partidul Social Democrat, prin Serviciul Contabilitate a constatat plata, la data de 08 octombrie 2018, a ultimei tranșe a contravalorii autovehiculului a dat catre Georgiana Drăghici."""
    
    # 'DOCUMENT' : """Prin cererea de chemare în judecată înregistrată pe rolul Curţii de Apel Bucureşti – Secţia a VIII- a Contencios Administrativ şi Fiscal sub numărul 2570/2/2017, reclamantul Curuti  Ionel, a solicitat, în contradictoriu cu pârâta Agenţia Naţională de Integritate anularea Raportului de evaluare nr. 9756/G/II/17.03.2017 întocmit de ANI - Inspecţia de Integritate şi obligarea pârâtei la plata cheltuielilor de judecata ocazionate.""",
    # 'DOCUMENT' : """S-au luat în examinare recursurile formulate de reclamanta S.C. Compania de Apă Târgoviște Dâmbovița S.A. și chemata în garanție S.C. Tadeco Consulting S.R.L. (fostă S.C. Fichtner Environment S.R.L.) împotriva Sentinţei nr. 97 din 12 aprilie 2017 pronunţată de Curtea de Apel Ploiești – Secţia a II-a Civilă, de Contencios Administrativ şi Fiscal. La apelul nominal, făcut în şedinţă publică, răspunde recurenta- reclamantă S.C. Compania de Apă Târgoviște Dâmbovița S.A., prin consilier juridic Niţă Vasile Laurenţiu, care depune delegaţie de reprezentare la dosar, recurenta - chemată în garanție S.C. Tadeco Consulting S.R.L. (fostă S.C. Fichtner Environment S.R.L.), prin consilier juridic Marinela Vladescu, care depune delegaţie de reprezentare la dosarul cauzei, lipsă fiind intimatul-pârât Ministerul Investiţiilor şi Proiectelor Europene (fostul Ministerul Fondurilor Europene). Procedura de citare este legal îndeplinită. Se prezintă referatul cauzei, magistratul – asistent învederând că recurenta-reclamantă a formulat o cerere de renunţare la cererea de chemare în judecată precum şi la cererea de chemare în garanţie, cu privire la care s-a depus punct de vedere în sensul de a se lua act de cererea de renunţare la judecată. Reclamanta S.C. Compania de Apă Târgoviște Dâmbovița S.A., prin avocat, conform art. 406 alin. 5 Cod procedură civilă, solicită a se lua act de cererea de renunţare la judecată, respectiv de chemare în garanţie, cu consecinţa anulării hotărârilor pronunţate de Curtea de Apel Ploieşti. Recurenta - chemată în garanție S.C. Tadeco Consulting S.R.L., prin consilier juridic, precizează că nu se opune renunţării la judecată, astfel cum a fost solicitată de recurenta-reclamantă, apreciind că sunt îndeplinite condiţiile prevăzute de dispozițiile art. 406 Cod procedură civilă.""",
    # 'DOCUMENT' : """Decizia nr. 12996 din 18.02.2016, înregistrata la Compania de Apa Târgovişte Dâmboviţa SA sub nr. 8260/23.02.2016 ce priveşte soluţionarea contestaţiei formulata de Compania de Apa Târgovişte Dâmboviţa SA împotriva notei de constatare a neregulilor si de stabilire a corecţiilor financiare nr.3966/19.01.2016; Nota de constatare a neregulilor a neregulilor si de stabilire a corecţiilor financiare nr. 3966 din 19.01.2016 înregistrata la Compania de Apa Târgovişte Dâmboviţa SA sub nr. 15178 din 30.04.201 5 şi Notificării cu privire la debit nr. 3968 din 19.01.2016 înregistrata la Compania de Apa Târgovişte Dâmboviţa SA sub nr. 3663/311-UIP din 22.01.2016.""",
    # 'DOCUMENT' : """Totodată, a notificat beneficiarii PNDL cu privire la epuizarea creditelor bugetare în proporţie de 80% pentru PNDL 1, ultimele transferuri efectuându-se parţial pentru solicitările de finanţare depuse până în data de 08.11.2017 (adresa nr. 155732/18.12.2017).""",
#     'DOCUMENT' : """S-au luat în examinare recursul formulat de petentul Lupea Nicodim Eugen împotriva sentinţei penale nr. 494 din data de 27 noiembrie 2020, pronunţate de Înalta Curte de Casaţie şi Justiţie – Secţia penală în dosarul nr. 3039/1/2020.
# La apelul nominal, făcut în şedinţă publică, a lipsit recurentul Lupea Eugen Nicodim.
# Procedura de citare a fost legal îndeplinită.
# În conformitate cu dispoziţiile art. 369 alin. 1 din Codul de procedură penală, instanţa a procedat la înregistrarea desfăşurării şedinţei de judecată cu mijloace tehnice, stocarea informaţiilor realizându-se în memoria calculatorului.
# S-a făcut referatul cauzei de către magistratul asistent, care a învederat următoarele:
# - cauza are ca obiect recursul formulat de petentul Lupea Nicodim împotriva sentinţei penale nr. 494 din data de 27 noiembrie 2020, pronunţate de Înalta Curte de Casaţie şi Justiţie – Secţia penală în dosarul nr. 3039/1/2020;""",
    # 'DOCUMENT' : """Cum în prezenta cauză s-a formulat contestaţie în anulare împotriva unei decizii prin care a fost respins, ca inadmisibil, recursul formulat de contestatorul Dumitrescu Iulian, cale de atac exercitată împotriva unei hotărâri, prin care au fost respinse, ca inadmisibile, căile de atac formulate de acelaşi contestator în nume propriu şi pentru numiţii Patatu Geta, Branzariu Maria Crina, Paltinisanu Adrian, Ignat Vasile, Ciurcu Octavian Constantin, Florici Gheorghe, Sfrijan Marius, Puscasu Ermina Nicoleta, Dragoi Silvia Alina, Bolog Sandrino Iulian, Popa Georgeta, Malanciuc Petru Iulian, Tudorei Vladimir, Buscu Nicoleta Cristina, Budai Paul, Lostun Elena, Bolohan Marcel şi Musca Marinela, împotriva încheierii penale nr. 226/RC din data de 7 iunie 2019, pronunțate de Înalta Curte de Casaţie şi Justiţie, Secţia penală, în dosarul nr. 1181/1/2019 , Completul de 5 Judecători, a constatat că prin hotărârea atacată nu a fost soluţionată""",
#     'DOCUMENT' : """S-a luat în examinare apelul formulat de contestatoarea Ignatenko-Păvăloiu Nela împotriva deciziei nr. 186/A din data de 6 iulie 2021, pronunţate de Înalta Curte de Casaţie şi Justiţie, Secţia penală, în dosarul nr. 1220/1/2021.
# La apelul nominal făcut în ședință publică, a lipsit apelanta contestatoare Ignatenko (Păvăloiu) Nela, pentru care a răspuns apărătorul ales, avocat Nastasiu Ciprian, cu împuternicire avocaţială la dosarul cauzei (fila 9 din dosar).
# Procedura de citare a fost legal îndeplinită.
# În conformitate cu dispozițiile art. 369 alin. (1) din Codul de procedură penală, s-a procedat la înregistrarea desfășurării ședinței de judecată cu mijloace tehnice, stocarea datelor realizându-se în memoria calculatorului.
# S-a făcut referatul cauzei de către magistratul asistent, care a învederat faptul că prezentul dosar are ca obiect apelul formulat de contestatoarea Ignatenko (Păvăloiu) Nela împotriva deciziei nr. 186/A din data de 6 iulie 2021, pronunţate de Înalta Curte de Casaţie şi Justiţie, Secţia penală, în dosarul nr. 1220/1/2021.
# Cu titlu prealabil, reprezentantul Ministerului Public a invocat excepţia inadmisibilităţii căii de atac formulate întrucât vizează o decizie definitivă, pronunţată de Secţia penală a instanţei supreme în calea extraordinară de atac a contestaţiei în anulare.
# Înalta Curte de Casaţie şi Justiţie, Completul de 5 Judecători a supus discuţiei părţilor excepţia inadmisibilităţii căii de atac, invocate de reprezentantul Ministerului Public.
# Apărătorul ales al apelantei contestatoare Ignatenko (Păvăloiu) Nela a învederat că este admisibil apelul declarat împotriva deciziei penale nr. 186/A din data de 6 iulie 2021, pronunţate de Înalta Curte de Casaţie şi Justiţie, Secţia penală, în dosarul nr. 1220/1/2021, această din urmă decizie fiind definitivă doar în ceea ce priveşte dispoziţia de admitere a contestaţiei în anulare formulate de S.C. Reciplia SRL.
# În ceea ce priveşte dispoziţia de respingere a contestaţiei în anulare formulate de contestatoarea Ignatenko (Păvăloiu) Nela, apărarea a opinat că în această ipoteză este aplicabil art. 432 alin. (4) din Codul de procedură penală, în cuprinsul căruia se stipulează faptul că sentinţele pronunţate în calea de atac a contestaţiei în anulare, altele decât cele privind admisibilitatea, sunt supuse apelului. În acest sens, apărătorul ales a făcut trimitere la decizia nr. 5/2015, pronunţată de instanţa supremă.""",
#     'DOCUMENT' : """Prin sentinţa penală nr. 112/F din data de 16 mai 2019 pronunţată în dosarul nr. 2555/2/2019 (1235/2019), Curtea de Apel Bucureşti – Secţia a II-a Penală, a admis sesizarea formulată de Parchetul de pe lângă Curtea de Apel Bucureşti.
# A dispus punerea în executare a mandatului european de arestare emis la 24.04.2019 de Procuratura Graz pe numele persoanei solicitate Buzdugan Eminescu (cetăţean român, fiul lui Giovani şi Monalisa, născut la data de 22.12.1996 în Foggia, Republica Italiană, CNP 1961222160087, domiciliat în mun. Drobeta Turnu Severin, str.Orly nr.13, judeţul Mehedinţi şi fără forme legale în mun. Craiova, str. Ştirbey Vodă nr.74, judeţul Dolj).""",
    # 'DOCUMENT' : """Verificând actele aflate la dosar, Înalta Curte constată că, la data de 03.05.2019 a fost înregistrată pe rolul instanţei, sub nr. 2555/2/2019 (1235/2019), sesizarea Parchetului de pe lângă Curtea de Apel Bucureşti, în conformitate cu dispoz. art. 101 alin. 4 din Legea nr. 302/2004 republicată, având ca obiect procedura de punere în executare a mandatului european de arestare emis la data de 24.04.2019 (fiind indicată din eroare data de 1.04.2019), de către autorităţile judiciare austriece, respectiv Procuratura Graz, pe numele persoanei solicitate BUZDUGAN EMINESCU, cetăţean român, fiul lui Giovani şi Monalisa, născut la data de 22.12.1996 în Foggia, Republica Italiană, CNP 1961222160087, domiciliat în mun. Drobeta Turnu Severin, str.Orly nr.13, judeţul Mehedinţi şi fără forme legale în mun. Craiova, str. Ştirbey Vodă nr.74, judeţul Dolj, urmărit pentru săvârşirea infracţiunii de  tentativă de furt prin efracţie într-o locuinţă, ca membru al unei organizaţii criminale,  prevăzute şi pedepsite de secţiunile 15, 127, 129/2 cifra 1, 130 alin. 1, al doilea caz şi alin. 2 din Codul penal austriac.""",
    # 'DOCUMENT' : """Mandatul european de arestare este o decizie judiciară emisă de autoritatea judiciară competentă a unui stat membru al Uniunii Europene, în speţă cea română, în vederea arestării şi predării către un alt stat membru, respectiv Austria, Procuratura Graz, a unei persoane solicitate, care se execută în baza principiului recunoașterii reciproce, în conformitate cu dispoziţiile Deciziei – cadru a Consiliului nr. 2002/584/JAI/13.06.2002, cât şi cu respectarea drepturilor fundamentale ale omului, aşa cum acestea sunt consacrate de art. 6 din Tratatul privind Uniunea Europeană.""",
    # 'DOCUMENT' : """Subsemnatul Damian Ionut Andrei, nascut la data 26.01.1976, domiciliat in Cluj, str. Cernauti, nr. 17-21, bl. J, parter, ap. 1 , declar pe propria raspundere ca sotia mea Andreea Damian, avand domiciliul flotant in Voluntari, str. Drumul Potcoavei nr 120, bl. B, sc. B, et. 1, ap 5B, avand CI cu CNP 1760126423013 nu detine averi ilicite.""",
        
    'DOCUMENT' : """Silviu Mihai si Silviu Mihail au mers impreuna la tribunal, la Sectia 2, sa se judece pe o bucata de pamanat din comuna Pantelimon, teren care apartine subsemnatului Pantelimon Marin-Ioan"""
      }
  
  res = eng.execute(inputs=test, counter=1)
  print(res)
