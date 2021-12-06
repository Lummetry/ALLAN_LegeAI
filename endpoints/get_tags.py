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



from libraries.model_server_v2 import FlaskWorker
from tagger.brain.emb_aproximator import EmbeddingApproximator

import constants as ct

_CONFIG = {
  'EMBGEN_MODEL' : '20211125_203842_embgen_model_sc_40_ep140.h5',
  'GENERATED_EMBEDS' : 'embgen_full_embeds.npy',
  'WORD_EMBEDS' : 'lai_embeddings_191K.pkl',
  'IDX2WORD' : 'lai_ro_i2w_191K.pkl'
  }

class GetTagsWorker(FlaskWorker):
  def __init__(self, **kwargs):
    super(GetTagsWorker, self).__init__(**kwargs)
    return
  
  def _load_model(self):
    fn_model = self.config_worker['EMBGEN_MODEL']
    fn_gen_emb = self.config_worker['GENERATED_EMBEDS']
    fn_emb = self.config_worker['WORD_EMBEDS']
    fn_i2w = self.config_worker['IDX2WORD']
    self.encoder = EmbeddingApproximator(log=self.log, fn_embeds=fn_emb, fn_idx2word=fn_i2w)
    self.encoder.setup_embgen_model(
      embgen_model_file=fn_model,
      generated_embeds_filename=fn_gen_emb,
      run_in_cpu=True
      )
    self._create_notification('LOAD', 'Loaded EmbeddingApproximator')
    return
    

  def _pre_process(self, inputs):
    doc = inputs['DOCUMENT']
    if len(doc) < ct.MODELS.TAG_MIN_INPUT:
      raise ValueError("Document: '{}' is below the minimum of {} words".format(
        doc, ct.MODELS.TAG_MIN_INPUT))
    embeds = self.encoder.encode(
      doc,
      direct_embeddings=True, 
      fixed_len=ct.MODELS.TAG_MAX_LEN, 
      raw_conversion=False,
      convert_unknown_words=True,
      )
    return embeds
    

  def _predict(self, prep_inputs):
    model = lambda x: x
    inputs = list(prep_inputs.shape)
    res = model(inputs)
    return res

  def _post_process(self, pred):
    res =  {'results' : pred}
    return res
    
    
if __name__ == '__main__':
  from libraries import Logger
  l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
  
