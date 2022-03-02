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
    

  def _pre_process(self, inputs):
    query = inputs['QUERY']

    if len(query.split(' ')) > ct.MODELS.QA_MAX_INPUT:
      raise ValueError("Query: '{}' exceedes max number of allowed words of {}".format(
        query, ct.MODELS.QA_MAX_INPUT))
    query_embeds = self.encoder.encode_convert_unknown_words(
      query,
      fixed_len=ct.MODELS.TAG_MAX_LEN
    )
    self.current_query_embeds = query_embeds # not needed in tagger

    n_hits = int(inputs.get('TOP_N', 10))
    
    
    docs = inputs['DOCUMENTS']
    self.current_docs_embeds = []
    
    for doc in docs:
        if len(doc) < ct.MODELS.TAG_MIN_INPUT:
          raise ValueError("Document: '{}' is below the minimum of {} words".format(
            doc, ct.MODELS.TAG_MIN_INPUT))
        doc_embeds = self.encoder.encode_convert_unknown_words(
          doc,
          fixed_len=ct.MODELS.TAG_MAX_LEN
        )
        self.current_docs_embeds.append(doc_embeds)
        
    print(self.current_docs_embeds)

    return query_embeds, n_hits
    

  def _predict(self, prep_inputs):
    inputs, n_hits = prep_inputs
    res = self.tagger_model(inputs).numpy()
    return res, n_hits

  def _post_process(self, pred):
    predictions, n_hits = pred
    top_n_idxs = np.argsort(predictions.squeeze())[-n_hits:]

    res = {}
    res['results'] = [
      [self.id_to_label[i], round(predictions.squeeze()[i].astype(float), 3)]
      for i in top_n_idxs
    ]
    res['results'].reverse()

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
          """decizia recurată a fost dată cu încălcarea autorităţii de lucru interpretat, respectiv cu încălcarea dispozitivului hotărârii preliminare pronunţate de Curtea de Justiţie a Uniunii Europene în Cauza C-52/07 (hotărâre care are autoritate de lucru interpretat „erga omnes”)"""
          ],
      'TOP_N' : 5
    },

    # {
    #   'QUERY' : 'Cat la suta din salariul brut merge la pensii pentru un programator?',
    #   'TOP_N' : 7
    # }
  ]

  for i,_input in enumerate(inputs_to_test):
    result = w.execute(inputs=_input, counter=i)
    log.P("Input: {}\nResult:{}".format(_input, result), color='m')
