# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 18:08:35 2022

@author: filip
"""

import numpy as np

from libraries.model_server_v2 import FlaskWorker
from tagger.brain.emb_aproximator import SimpleEmbeddingApproximatorWrapper

import constants as ct

_CONFIG = {
  'TAGGER_MODEL': '20211206_205159_ep35_R0.61_P0.90_F10.73.h5',
  'LABEL2ID': 'dict_lbl_37.pkl',
  'EMBGEN_MODEL' : '20211125_203842_embgen_model_sc_40_ep140.h5',
  'GENERATED_EMBEDS' : 'embgen_full_embeds.npy',
  'WORD_EMBEDS' : 'lai_embeddings_191K.pkl',
  'IDX2WORD' : 'lai_ro_i2w_191K.pkl'
  }

MIN_SUBDOCUMENTS = 2
MIN_SUBDOCUMENT_WORDS = 100
MAX_QUERY_WORDS = 50

MAX_COS_DISTANCE = 0.5


__VER__='0.1.0.0'
class GetMarkWorker(FlaskWorker):
  def __init__(self, **kwargs):
    super(GetMarkWorker, self).__init__(**kwargs)
    return
  
  def _load_model(self):
    fn_tagger_model = self.config_worker['TAGGER_MODEL']
    fn_model = self.config_worker['EMBGEN_MODEL']
    fn_gen_emb = self.config_worker['GENERATED_EMBEDS']
    fn_emb = self.config_worker['WORD_EMBEDS']
    fn_i2w = self.config_worker['IDX2WORD']
    fn_label_to_id = self.config_worker['LABEL2ID']

    self.label_to_id = self.log.load_pickle_from_data(fn_label_to_id)
    self.id_to_label = {v: k for k, v in self.label_to_id.items()}
    self.tagger_model = self.log.load_keras_model(fn_tagger_model)

    self.encoder = SimpleEmbeddingApproximatorWrapper(
      log=self.log,
      fn_embeds=fn_emb,
      fn_idx2word=fn_i2w,
      embgen_model_file=fn_model,
      generated_embeds_filename=fn_gen_emb,
    )

    warmup_input = self.encoder.encode_convert_unknown_words(
      "Warmup",
      fixed_len=ct.MODELS.TAG_MAX_LEN
    )
    self.tagger_model(warmup_input)

    self._create_notification('LOAD', 'Loaded EmbeddingApproximator')
    return


#######
# AUX #
#######

  def cosine_distance(self, a, b):
      a_norm = np.linalg.norm(a)
      b_norm = np.linalg.norm(b)
      similiarity = np.dot(a, b.T)/(a_norm * b_norm)
      dist = 1. - similiarity
      
      return dist

    

  def _pre_process(self, inputs): 
      
    # Read query
    query = inputs['QUERY']
    if len(query.split(' ')) > MAX_QUERY_WORDS:
        raise ValueError("Query: '{}' exceedes max number of allowed words of {}".format(
            query, MAX_QUERY_WORDS))
    # Embed query
    query_embeds = self.encoder.encode_convert_unknown_words(
              query,
              fixed_len=ct.MODELS.TAG_MAX_LEN
    )    
    
    # Read subdocument list
    docs = inputs['DOCUMENTS']    
    if len(docs) < MIN_SUBDOCUMENTS:
      raise ValueError("Number of subdocuments is below the minimum of {}".format(
        MIN_SUBDOCUMENTS))
    
    # Embed each subdocument
    docs_embeds = []    
    for doc in docs:
        if len(doc) < MIN_SUBDOCUMENT_WORDS:
          raise ValueError("Document: '{}' is below the minimum of {} words".format(
            doc, MIN_SUBDOCUMENT_WORDS))
        doc_embeds = self.encoder.encode_convert_unknown_words(
          doc,
          fixed_len=ct.MODELS.TAG_MAX_LEN
        )
        docs_embeds.append(doc_embeds)

    return query_embeds, docs_embeds    


  def _predict(self, prep_inputs):
    query_embeds, docs_embeds = prep_inputs
    
    query_tag_vector = self.tagger_model(query_embeds).numpy().squeeze()
    
    selected_docs = []
    for i, doc_embeds in enumerate(docs_embeds):
        doc_tag_vector = self.tagger_model(doc_embeds).numpy().squeeze()
        distance = self.cosine_distance(query_tag_vector, doc_tag_vector)
        
        if distance < MAX_COS_DISTANCE:
            selected_docs.append(i)    
    
    return selected_docs

  def _post_process(self, pred):
    selected_docs = pred
    
    # TODO REPLACE N HITS

    res = {}
    res['results'] = selected_docs

    return res
    
    
if __name__ == '__main__':
  from libraries import Logger
  log = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
  w = GetMarkWorker(log=log, default_config=_CONFIG, verbosity_level=1)

  inputs_to_test = [
    {
      'QUERY' : 'Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti?',
      'DOCUMENTS': [
          """Subsemnatul Damian Ionut Andrei, domiciliat in Cluj, Strada Cernauti, nr. 17-21, bl. J, parter, ap. 1 , nascut pe data 24-01-1982, declar pe propria raspundere ca sotia mea Andreea Damian, avand domiciliul flotant in Bucuresti, str. Drumul Potcoavei nr 120, bl. B, sc. B, et. 1, ap 5B, avand CI cu CNP 1760126413223 serie RK, numar 897567 nu detine averi ilicite""",
          """decizia recurată a fost dată cu încălcarea autorităţii de lucru interpretat, respectiv cu încălcarea dispozitivului hotărârii preliminare pronunţate de Curtea de Justiţie a Uniunii Europene în Cauza C-52/07 (hotărâre care are autoritate de lucru interpretat „erga omnes”)""",
           """Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti? Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti? Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti?""",
          ]
    },
  ]

  for i,_input in enumerate(inputs_to_test):
    result = w.execute(inputs=_input, counter=i)
    print(result)
