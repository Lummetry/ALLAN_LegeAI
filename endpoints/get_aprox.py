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
from utils.utils import simple_levenshtein_distance
from tagger.brain.emb_aproximator import EmbeddingApproximator

_CONFIG = {
  'EMBGEN_MODEL' : '20211125_203842_embgen_model_sc_40_ep140.h5',
  'GENERATED_EMBEDS' : 'embgen_full_embeds.npy',
  'WORD_EMBEDS' : 'lai_embeddings_191K.pkl',
  'IDX2WORD' : 'lai_ro_i2w_191K.pkl'
  }

class GetAproxWorker(FlaskWorker):
  def __init__(self, **kwargs):
    super(GetAproxWorker, self).__init__(**kwargs)
    return
  
  def _load_model(self):
    fn_model = self.config_worker['EMBGEN_MODEL']
    fn_gen_emb = self.config_worker['GENERATED_EMBEDS']
    fn_emb = self.config_worker['WORD_EMBEDS']
    fn_i2w = self.config_worker['IDX2WORD']
    self.eng = EmbeddingApproximator(log=self.log, fn_embeds=fn_emb, fn_idx2word=fn_i2w)
    self.eng.setup_embgen_model(
      embgen_model_file=fn_model,
      generated_embeds_filename=fn_gen_emb,
      run_in_cpu=True
      )
    self._create_notification('LOAD', 'Loaded EmbeddingApproximator')
    return
    

  def _pre_process(self, inputs):
    word = inputs['query']
    if word in self.eng.dic_word2index:
      raise ValueError("Presumed unknown word '{}' already is in existing vacabulary".format(word))
    top_n = inputs.get('top_n', 1)
    word = word.lower() # this is important as the model is lowercase biased
    return word, top_n
    

  def _predict(self, prep_inputs):
    word, top_n = prep_inputs
    res = self.eng.get_unk_word_similar_word(word, top=top_n) 
    return res

  def _post_process(self, pred):
    res =  {'results' : pred}
    return res
    
    
if __name__ == '__main__':
  from libraries import Logger
  l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
  eng = GetAproxWorker(log=l, default_config=_CONFIG, verbosity_level=1)
  eng._load_model()
  print(eng.eng.get_unk_word_similar_word('tva', top=10, debug=True))
  
