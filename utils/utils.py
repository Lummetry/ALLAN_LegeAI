# -*- coding: utf-8 -*-
"""
Copyright 2019-2021 Lummetry.AI (Knowledge Investment Group SRL). All Rights Reserved.
* NOTICE:  All information contained herein is, and remains
* the property of Knowledge Investment Group SRL.
* The intellectual and technical concepts contained
* herein are proprietary to Knowledge Investment Group SRL
* and may be covered by Romanian and Foreign Patents,
* patents in process, and are protected by trade secret or copyright law.
* Dissemination of this information or reproduction of this material
* is strictly forbidden unless prior written permission is obtained
* from Knowledge Investment Group SRL.

@copyright: Lummetry.AI
@author: Lummetry.AI - AID
@project:
@description:

"""

from difflib import ndiff
import re
from gensim import utils
import numpy as np
import spacy

HTML_TAG_CLEANER = re.compile('<.*?>')


def raw_text_to_words(text, max_len=20):
  clean_line = re.sub(HTML_TAG_CLEANER, '', text)
  preprocessed = utils.simple_preprocess(clean_line, deacc=True, max_len=20)
  return preprocessed

def clean_words_list(lst_words, max_len=20):
  lst_new = []
  for word in lst_words:
    preprocessed = utils.simple_preprocess(word, deacc=True, max_len=20)
    new_word = "_".join(preprocessed)
    lst_new.append(new_word)
  return lst_new


def simple_levenshtein_distance(reference, hypothesis, normalize=True):
  str1 = reference
  str2 = hypothesis
  counter = {"+": 0, "-": 0}
  distance = 0
  for edit_code, *_ in ndiff(str1, str2):
      if edit_code == " ":
          distance += max(counter.values())
          counter = {"+": 0, "-": 0}
      else: 
          counter[edit_code] += 1
  distance += max(counter.values())
  if normalize:
    distance /= len(reference)
  return distance

def test_model(log, 
               model, 
               words=['leafa', 'salarizare', 'impozit', 'tva', 'divort', 'munca', 'frauda', 'copil'], 
               name=None,
               topn=5,
               color='b',
               ):
  if name is not None:
    log.P("Test for model '{}' {}".format(name, ' ' * 20), color=color)
  for _w in words:
    res = model.wv.most_similar(_w, topn=topn)
    log.P("  Top {} for '{}': {}".format(
      topn, _w,
      ", ".join(["{}: {:.2f}".format(_sw, _sc) for _sw, _sc in res ])
      ), color=color)
  if name is not None:
    log.P("*" * 80)
  return


def K_triplet_loss(y_pred, beta=0.5,):
   import tensorflow.keras.backend as K

   anchor = y_pred[0]
   positive = y_pred[1]
   negative = y_pred[2]

   similar_dist = K.sum(K.square(anchor - positive), axis=1)
   diff_dist = K.sum(K.square(anchor - negative), axis=1)
   loss = K.maximum(similar_dist - diff_dist + beta, 0.0)
   loss = K.expand_dims(loss)
   return loss

def K_identity_loss(y_true, y_pred):
  import tensorflow.keras.backend as K
  return K.mean(y_pred - 0 * y_true)


#######################
# TITLE PREPROCESSING #
#######################

REMOVE_PARAN = 0
REMOVE_PREFIX = 1
REMOVE_POS = 2
REMOVE_STOPWORDS = 3
REMOVE_DEP = 4
REMOVE_NONALPHA = 5
REMOVE_ENTITIES = 6

TITLE_STOPWORDS = ['nr', 'nr.', 'art', 'art.', 'lit', 'lit.']
TITLE_PREFIX = ['sentinţă', 'sentinta', 'decizia', 'decizie', 'decretul', 'decret', 'ordinul',
               'hotărârea', 'hotararea', 'actul', 'cauza']

def preprocess_title(title, nlp=None, 
                     proc=[REMOVE_PARAN, REMOVE_PREFIX, REMOVE_POS, REMOVE_STOPWORDS,
                           REMOVE_DEP, REMOVE_NONALPHA, REMOVE_ENTITIES], 
                     debug=False
                    ):
    """ Preprocess title using several techniques and heuristics. """
    
    if nlp is None:
        SPACY_MODEL = 'ro_core_news_lg'
        try:
            nlp = spacy.load(SPACY_MODEL)
        except OSError:
            spacy.cli.download(SPACY_MODEL)
            nlp = spacy.load(SPACY_MODEL) 
    
    doc = nlp(title)
    remove_list = np.zeros(len(doc))
    
    paran = False
    prefix = False
    
    for i, tok in enumerate(doc):
        
        if debug:
            print(tok, tok.pos_, tok.dep_, tok.is_alpha)
        
        # Remove prefixes
        if i == 0 and tok.text.lower() in TITLE_PREFIX:
            prefix = True
            remove_list[i] = 1
        elif prefix == True:
            if tok.pos_ == 'NUM' or tok.dep_ == 'nummod':
                prefix = False
            remove_list[i] = 1
        
        # Remove by POS
        if (tok.pos_ in ['PUNCT', 'SPACE', 'SYM', 'X'] or
            # Remove punctuation, symbols, other
            
            tok.pos_ in ['ADP', 'CCONJ', 'SCONJ', 'DET'] or
            # Remove preopositions, conjunctions, determiners
            
            tok.pos_ in ['NUM']           
            # Remove numerals
           
           ):
            remove_list[i] = 1
            
        # Remove parantheses
        if REMOVE_PARAN in proc:
                        
            if tok.text in ['(', '[', '{']:
                print(tok)
                paran = True
                remove_list[i] = 1
                
            elif paran == True:
                
                if tok.text in [')', ']', '}']:
                    paran = False
                    
                remove_list[i] = 1
            
        # Remove certain stopwords
        if (tok.is_stop or 
            tok.text in TITLE_STOPWORDS):
            remove_list[i] = 1
            
        # Remove by dependency
        if tok.dep_ == 'nummod':
            remove_list[i] = 1  
            
        # Remove non alphabetic
        if tok.is_alpha == False:
            remove_list[i] = 1
            
    # Remove entities
    if REMOVE_ENTITIES in proc:
        for ent in doc.ents:

            if ent.label_ in ['PERSON', 'LOC', 'GPE']:
                for j in range(ent.start, ent.end):
                    if doc[i].is_title:
                        # Only remove tokens with starting with capital letters
                        remove_list[j] = 1       
                
            if ent.label_ in ['NUMERIC_VALUE']:
                for j in range(ent.start, ent.end):
                    remove_list[j] = 1
    
                
    # Build output title
    new_title = [doc[i].text for i in range(len(doc)) if remove_list[i] == 0]
    new_title = ' '.join(new_title)
                
    return new_title

if __name__ == '__main__':
    text = """"trafic de droguri", prev. de art. 2 al. 1 si 2 din Legea nr. 143/2000 cu aplicarea art. 41 al. 2 Cod penal si "detinere de droguri de mare risc in vederea consumului propriu", prev de art. 4 al. 1 si 2 din Legea nr. 143/2000;
"""
    res = preprocess_title(text, proc=[REMOVE_PARAN, REMOVE_PREFIX, REMOVE_POS, REMOVE_STOPWORDS,
                                       REMOVE_DEP, REMOVE_NONALPHA, REMOVE_ENTITIES])
    print(res)