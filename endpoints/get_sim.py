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

from gensim.models import Word2Vec

from libraries.model_server_v2 import FlaskWorker
from utils.utils import simple_levenshtein_distance

_CONFIG = {
  'MODEL' : '20211124_082955_e128_v191K_final'
  }

class GetSimWorker(FlaskWorker):
  def __init__(self, **kwargs):
    super(GetSimWorker, self).__init__(**kwargs)
    return
  
  def _load_model(self):
    """
    See docstring in parent
    Abstract method implementation:
      - in this case, we have to deal with a Word2Vec model and we load it in memory
    """
    model_fn = self.log.get_models_file(self.config_worker['MODEL'])
    self.model = Word2Vec.load(model_fn)
    self._create_notification('LOAD', 'Loaded model {}'.format(model_fn))
    return
    

  def _pre_process(self, inputs):
    """
    See docstring in parent
    Abstract method implementation:
      - parses the request inputs and keep the value for 'QUERY' and 'TOP_N'
      - custom verification
    """
    word = inputs['QUERY']
    n_hits = int(inputs.get('TOP_N', 5))
    self.__include_raw = inputs.get('raw', False)
    if word not in self.model.wv:
      raise ValueError("Word '{}' not found in database".format(word))
    if n_hits > 10:
      raise ValueError("top_n max value is 10, received {}".format(n_hits))
    return word, n_hits

  def _predict(self, prep_inputs):
    """
    See docstring in parent
    Abstract method implementation:
      - calls the model to get most similar words
      - custom logic based on lavenshtein_distance to discard words that LOOK! similar.
    """
    word, n_hits = prep_inputs
    lst_words = self.model.wv.most_similar(word, topn=100)
    lst_raw = self.model.wv.most_similar(word, topn=n_hits)
    lst_proposed = []
    for proposed, dist in lst_words:
      dist = simple_levenshtein_distance(word, proposed, normalize=False)
      if dist > 1:
        lst_proposed.append(proposed)
      if len(lst_proposed) >= n_hits:
        break
    return lst_proposed, lst_raw

  def _post_process(self, pred):
    """
    See docsting in parent
    Abstract method implementation:
      - packs the predictio for the end-user
    """
    res =  {'results' : pred[0]}
    if self.__include_raw:
      res['raw_results'] = pred[1]
    return res
    
    
if __name__ == '__main__':
  from libraries import Logger
  l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
  model_fn = l.get_models_file('20211124_082955_e128_v191K_final')
  model = Word2Vec.load(model_fn)
