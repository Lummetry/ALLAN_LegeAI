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
@author: Lummetry.AI
@project: 
@description:
@created on: Fri Dec  3 08:59:47 2021
@created by: damia
"""

import numpy as np

from libraries.model_server_v2 import FlaskWorker
from tagger.brain.emb_aproximator import EmbeddingApproximator

import constants as ct

_CONFIG = {
  'TAGGER_MODEL': '20211206_205159_ep35_R0.61_P0.90_F10.73.h5',
  'LABEL2ID': 'dict_lbl_37.pkl',
  'EMBGEN_MODEL' : '20211125_203842_embgen_model_sc_40_ep140.h5',
  'GENERATED_EMBEDS' : 'embgen_full_embeds.npy',
  'WORD_EMBEDS' : 'lai_embeddings_191K.pkl',
  'IDX2WORD' : 'lai_ro_i2w_191K.pkl'
  }

class GetQAWorker(FlaskWorker):
  def __init__(self, **kwargs):
    super(GetQAWorker, self).__init__(**kwargs)
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
    self.encoder = EmbeddingApproximator(log=self.log, fn_embeds=fn_emb, fn_idx2word=fn_i2w)
    self.encoder.setup_embgen_model(
      embgen_model_file=fn_model,
      generated_embeds_filename=fn_gen_emb,
      run_in_cpu=True
      )

    warmup_input = self.encoder.encode(
      "Warmup",
      direct_embeddings=True,
      fixed_len=ct.MODELS.TAG_MAX_LEN,
      raw_conversion=False,
      convert_unknown_words=True,
    )
    self.tagger_model(warmup_input)

    self._create_notification('LOAD', 'Loaded EmbeddingApproximator')
    return
    

  def _pre_process(self, inputs):
    query = inputs['QUERY']

    if len(query.split(' ')) > ct.MODELS.QA_MAX_INPUT:
      raise ValueError("Query: '{}' exceedes max number of allowed words of {}".format(
        query, ct.MODELS.QA_MAX_INPUT))
    embeds = self.encoder.encode(
      query,
      direct_embeddings=True, 
      fixed_len=ct.MODELS.TAG_MAX_LEN, 
      raw_conversion=False,
      convert_unknown_words=True,
    )
    self.current_query_embeds = embeds # not needed in tagger

    n_hits = int(inputs.get('TOP_N', 10))

    return embeds, n_hits
    

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
  l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
  eng = GetQAWorker(log=l, default_config=_CONFIG, verbosity_level=1)
  eng._load_model()
  
