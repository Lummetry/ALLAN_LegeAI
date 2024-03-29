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
from transformers import AutoTokenizer, TFBertModel
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf



import constants as ct

_CONFIG = {
  
}


class GetTagsWorker(FlaskWorker):
  def __init__(self, **kwargs):
    super(GetTagsWorker, self).__init__(**kwargs)
    return

  def _load_model(self):
    fn_tagger_model = self.config_worker['TAGGER_MODEL']
    fn_label_to_id = self.config_worker['LABEL2ID']
    fn_bert_backbone = self.config_worker["BERT_BACKBONE"]
    fn_bert_max_seq_len = self.config_worker["BERT_MAX_SEQ_LEN"]

    self.label_to_id = self.log.load_pickle_from_data(fn_label_to_id)
    self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    bert_backbone =  TFBertModel.from_pretrained(fn_bert_backbone)
    self.tokenizer = AutoTokenizer.from_pretrained(fn_bert_backbone)

    input_ids      = layers.Input(shape=(fn_bert_max_seq_len,), dtype='int64', name="input_ids")
    attention_mask = layers.Input(shape=(fn_bert_max_seq_len,), dtype='int32', name="attention_mask")
    bert_layer = bert_backbone(input_ids, attention_mask)[0]
    # get cls output
    bert_output = layers.Lambda(lambda seq: seq[:, 0, :])(bert_layer)
    # add dropout?
    classifier_layer = layers.Dense(len(self.label_to_id), activation="sigmoid")(bert_output)

    model = keras.Model(inputs=(input_ids, attention_mask), outputs=classifier_layer)
    model.build(input_shape=[(None, fn_bert_max_seq_len), (None, fn_bert_max_seq_len)])
    model.compile()

    model.load_weights(fn_tagger_model)

    self.tagger_model = model

    return

  def _pre_process(self, inputs):
    doc = inputs['DOCUMENT']
    n_hits = int(inputs.get('TOP_N', 10))

    inputs = self.tokenizer([doc], padding="max_length", truncation=True, max_length=self.config_worker["BERT_MAX_SEQ_LEN"], is_split_into_words=False)

    return inputs, n_hits

  def _predict(self, prep_inputs):
    inputs, n_hits = prep_inputs
    inputs = [inputs["input_ids"], inputs["attention_mask"]]
    tf_dataset = tf.data.Dataset.from_tensor_slices((inputs[0], inputs[1])).batch(1)
    for x in tf_dataset:
      res = self.tagger_model.predict(x)[0]

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

  l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=True)

  a = GetTagsWorker(log=l, default_config=_CONFIG, verbosity_level=0)
  a._load_model()
  ins = a._pre_process({"DOCUMENT": "Art. 63. - (1) Aerodromurile civile certificate din România pentru care există obligaţia realizării hărţilor strategice de zgomot şi a planurilor de acţiune aferente, precum şi criteriile care stau la baza realizării acestora, sunt stabilite prin hotărâre a Guvernului, la iniţiativa autorităţii publice centrale pentru protecţia mediului. (2) Administratorii aerodromurilor civile certificate din România prevăzute la alin. (1) au obligaţia ca, atunci când elaborează programe de dezvoltare a infrastructurii de aerodrom, să utilizeze cartarea zgomotului în vederea previzionării impactului activităţilor viitoare asupra vecinătăţilor aerodromurilor. (3) Autorităţile administraţiei publice locale au obligaţia integrării în documentaţiile de urbanism şi de amenajare a teritoriului a hărţilor strategice de zgomot prevăzute la alin. (1), precum şi a programelor de dezvoltare specificate la alin. (2), împreună cu rezultatele cartării zgomotului. (4) Metodologia de cartare a zgomotului în vederea previzionării impactului activităţilor existente şi viitoare de pe un aerodrom asupra vecinătăţilor acestuia, se aprobă prin hotărâre a Guvernului, la iniţiativa autorităţii publice centrale pentru protecţia mediului."})
  p = a._predict(ins)
  r = a._post_process(p)
  

  print(r)