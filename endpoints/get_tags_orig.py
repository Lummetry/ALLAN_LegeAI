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

import numpy as np

from libraries.model_server_v2 import FlaskWorker
from tagger.brain.emb_aproximator import EmbeddingApproximator, SimpleEmbeddingApproximatorWrapper

import constants as ct

_CONFIG = {
  'TAGGER_MODEL': '20211206_205159_ep35_R0.61_P0.90_F10.73.h5',
  'LABEL2ID': 'dict_lbl_37.pkl',
  'EMBGEN_MODEL': '20211125_203842_embgen_model_sc_40_ep140.h5',
  'GENERATED_EMBEDS': 'embgen_full_embeds.npy',
  'WORD_EMBEDS': 'lai_embeddings_191K.pkl',
  'IDX2WORD': 'lai_ro_i2w_191K.pkl'
}


class GetTagsWorkerOrig(FlaskWorker):
  def __init__(self, **kwargs):
    super(GetTagsWorkerOrig, self).__init__(**kwargs)
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
    doc = inputs['DOCUMENT']
    if len(doc) < ct.MODELS.TAG_MIN_INPUT:
      raise ValueError("Document: '{}' is below the minimum of {} words".format(
        doc, ct.MODELS.TAG_MIN_INPUT))
    embeds = self.encoder.encode_convert_unknown_words(
      doc,
      fixed_len=ct.MODELS.TAG_MAX_LEN
    )
    self.current_query_embeds = embeds

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
  eng = GetTagsWorkerOrig(log=l, default_config=_CONFIG, verbosity_level=1)
  
  _input = {
      # 'DOCUMENT': """Subsemnatul Damian Ionut Andrei, domiciliat in Cluj, Strada Cernauti, nr. 17-21, bl. J, parter, ap. 1 , nascut pe data 24-01-1982, declar pe propria raspundere ca sotia mea Andreea Damian, avand domiciliul flotant in Bucuresti, str. Drumul Potcoavei nr 120, bl. B, sc. B, et. 1, ap 5B, avand CI cu CNP 1760126413223 serie RK, numar 897567 nu detine averi ilicite""",
      'DOCUMENT': """decizia recurată a fost dată cu încălcarea autorităţii de lucru interpretat, respectiv cu încălcarea dispozitivului hotărârii preliminare pronunţate de Curtea de Justiţie a Uniunii Europene în Cauza C-52/07 (hotărâre care are autoritate de lucru interpretat „erga omnes”)""",
          
      'TOP_N' : 5
    }

  result = eng.execute(inputs=_input, counter=1)
  print(result)

