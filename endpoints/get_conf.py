# -*- coding: utf-8 -*-

from libraries.model_server_v2 import FlaskWorker

import constants as ct
import numpy as np
import spacy
import re


_CONFIG = {
  'SPACY_MODEL' : 'ro_core_news_md',
  'DEBUG' : True,
  'PRINT_RESULT': True
 }


CNP_REG1 = re.compile(r'[0-9]{13}')
CNP_REG2 = re.compile(r'[1-8][0-9]{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[1-2][0-9]|3[0-1])[0-9]{6}')

CNP_NO_CHECK = 1
CNP_BASIC_CHECK = 2
CNP_FULL_CHECK = 3

MATCHED_NERS = {'PERSON', 'LOC'}

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
    
    def find_match(self, match, text, res):
        """ Find the position of a match """
        
        pos = text.find(match)
        
        # If the same match already exists, look for the next one
        while pos in res:
            pos = text.find(match, pos + 1)
                            
        return pos
    
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
                res[self.find_match(match, text, res)] = 'CNP'
                
        return res
    
    def match_ner(self, nlp, text,
                  ners=MATCHED_NERS
                 ):
        """ Return the position of spaCy named entities in a text. """
        matches = {}    
        
        doc = nlp(text)
        
        for ent in doc.ents:
            if ent.label_ in ners:
                # matches[ent.start_char] = ent.label_
                matches[ent.start_char] = ent.text
                
        return matches   
    
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
        
        # Match NERs
        # matches.update(self.match_ner(self.nlp_model, doc))    
              
        return matches

    def _post_process(self, pred):
        
        matches = pred
        idxs = list(matches.keys())
        
        res = {}
        res['results'] = idxs
        
        return res


if __name__ == '__main__':
  from libraries import Logger

  l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
  eng = GetConfWorker(log=l, default_config=_CONFIG, verbosity_level=1)
  
  test = {
        # 'DOCUMENT': """Un contribuabil al cărui cod numeric personal este 2548016600768 va completa caseta "Cod fiscal" astfel:""",
      
        'DOCUMENT': """Se desemnează domnul Cocea Radu, avocat, domiciliat în municipiul Bucureşti, Bd. Laminorului nr. 84, sectorul 1, legitimat cu C.I. seria RD nr. 040958, eliberată la data de 16 septembrie 1998 de Secţia 5 Poliţie Bucureşti, CNP 1561119034963, în calitate de administrator special.""", 
        
        # 'DOCUMENT': """Cod numeric personal: 1505952103022. Doi copii minori înregistraţi în documentul de identitate.""",
        
        # 'DOCUMENT': """Bătrîn Cantemhir-Marian, porcine, Str. Cardos Iacob nr. 34, Arad, judeţul Arad, 1850810020101. 
        # Almăjanu Steliana, porcine, Comuna Peretu, judeţul Teleorman, 2580925341708.""",
        
      }
  
  res = eng.execute(inputs=test, counter=1)
  print(res)
