# -*- coding: utf-8 -*-
"""
Copyright 2019-2021 Lummetry.AI (4E SOFTWARE SRL). All Rights Reserved.


* NOTICE:  All information contained herein is, and remains the property of 4E SOFTWARE SRL.  
* The intellectual and technical concepts contained herein are proprietary to 4E SOFTWARE SRL
* and may be covered by Romanian and Foreign Patents, patents in process, and are protected 
* by trade secret or copyright law.
* Dissemination of this information or reproduction of this material is strictly forbidden 
* unless prior written permission is obtained from 4E SOFTWARE SRL.
*
*
*  RO:
*    Modul software TempRent, proiect finanțat în cadrul POC, Axa prioritara 2 - Tehnologia Informației și Comunicațiilor (TIC) 
*    pentru o economie digitală competitivă, Prioritatea de investiții 2b - Dezvoltarea produselor și s
*    erviciilor TIC, a comerțului electronic și a cererii de TIC, cod SMIS 142474, 
*    Contractul de finanțare nr. 2/221_ap3/24.06.2021.
*

  
  RO:
    Acest produs a fost livrat si realizat in baza serviciilor de cercetare-inovare industrială 
    conform contract de servicii nr. 9 din 01.11.2021 folosind modulele AI "ALLAN" aferente "TempRent" -  
    Proiect finanțat în cadrul POC, Axa prioritara 2 - Tehnologia Informației și Comunicațiilor (TIC) 
    pentru o economie digitală competitivă, Prioritatea de investiții 2b - Dezvoltarea produselor și s
    erviciilor TIC, a comerțului electronic și a cererii de TIC, cod SMIS 142474, 
    Contractul de finanțare nr. 2/221_ap3/24.06.2021.
    
"""



from libraries.model_server_v2 import FlaskWorker
from utils.utils import simple_levenshtein_distance
from tagger.brain.emb_aproximator import EmbeddingApproximator, SimpleEmbeddingApproximatorWrapper

_CONFIG = {
  'EMBGEN_MODEL' : '20211125_203842_embgen_model_sc_40_ep140.h5',
  'GENERATED_EMBEDS' : 'embgen_full_embeds.npy',
  'WORD_EMBEDS' : 'lai_embeddings_191K.pkl',
  'IDX2WORD' : 'lai_ro_i2w_191K.pkl'
  }

class GetAproxWorker(FlaskWorker):
  """
  Implementation of the worker for GET_APPROX endpoit;
  - as the worker runs on thread, then no prints are allowed; use `_create_notification` and see all the notifications
    when calling /notifications of the server.
  """

  def __init__(self, **kwargs):
    super(GetAproxWorker, self).__init__(**kwargs)
    return
  
  def _load_model(self):
    """
    See docstring in parent
    Abstract method implementation:
      - in this case, `EmbeddingApproximator` is our model
    """
    fn_model = self.config_worker['EMBGEN_MODEL']
    fn_gen_emb = self.config_worker['GENERATED_EMBEDS']
    fn_emb = self.config_worker['WORD_EMBEDS']
    fn_i2w = self.config_worker['IDX2WORD']
    self.eng = SimpleEmbeddingApproximatorWrapper(
      log=self.log,
      embgen_model_file=fn_model,
      generated_embeds_filename=fn_gen_emb,
      fn_idx2word=fn_i2w,
      fn_embeds=fn_emb
    )
    self._create_notification('LOAD', 'Loaded EmbeddingApproximator')
    return
    

  def _pre_process(self, inputs):
    """
    See docstring in parent
    Abstract method implementation:
      - parses the request inputs and keep the value for 'QUERY' (with some custom logic - lower()) and 'TOP_N'
    """
    word = inputs['QUERY']
    if word in self.eng.dic_word2index:
      raise ValueError("Presumed unknown word '{}' already is in existing vacabulary".format(word))
    top_n = int(inputs.get('TOP_N', 1))
    word = word.lower() # this is important as the model is lowercase biased
    return word, top_n

  def _predict(self, prep_inputs):
    """
    See docstring in parent
    Abstract method implementation:
      - calls the model to get aprox words
    """
    word, top_n = prep_inputs
    res = self.eng.get_unk_word_similar_word(word, top=top_n) 
    return res

  def _post_process(self, pred):
    """
    See docsting in parent
    Abstract method implementation:
      - packs the predictio for the end-user
    """
    res =  {'results' : pred}
    return res
    
    
if __name__ == '__main__':
  from libraries import Logger
  l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
  eng = GetAproxWorker(log=l, default_config=_CONFIG, verbosity_level=1)
  eng._load_model()
  print(eng.eng.get_unk_word_similar_word('tva', top=10))
  
