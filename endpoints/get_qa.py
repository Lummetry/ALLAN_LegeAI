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
@created by: mihai.masala
"""

import numpy as np

from libraries.model_server_v2 import FlaskWorker
from transformers import AutoTokenizer, TFBertModel
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf
import csv



import constants as ct

_CONFIG = {

}


class GetQAWorker(FlaskWorker):
  def __init__(self, **kwargs):
    super(GetQAWorker, self).__init__(**kwargs)
    return

  def _load_model(self):
    fn_tagger_model = self.config_worker['TAGGER_MODEL']
    fn_label_to_id = self.config_worker['LABEL2ID']
    fn_bert_backbone = self.config_worker["BERT_BACKBONE"]
    fn_bert_max_seq_len = self.config_worker["BERT_MAX_SEQ_LEN"]
    fn_mapping = self.config_worker["MAPPING_FILE"]

    self.label_to_id = self.log.load_pickle_from_data(fn_label_to_id)
    self.id_to_label = {v: k for k, v in self.label_to_id.items()}

    self.mappings = {}
    with open(fn_mapping) as csv_file:
      csv_reader = csv.reader(csv_file, delimiter=',')
      for row_index, row in enumerate(csv_reader):
          if row_index == 0:
            continue
          else:
              tags_label = str(row[0])
              qa_labels = str(row[1]).split(",")
              qa_labels = list(map(lambda x: x.strip(), qa_labels))
              qa_labels = list(filter(lambda x: x != "", qa_labels))
              self.mappings[tags_label] = qa_labels
    
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
    doc = inputs['QUERY']
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
    top_n_idxs = top_n_idxs[::-1]


    # res = {}
    # res['results'] = [
    #   [self.id_to_label[top_n_index], round(predictions.squeeze()[top_n_index].astype(float), 3)] for top_n_index in top_n_idxs
    # ]

    res = {}
    res['results'] = []
    for top_n_id in top_n_idxs:
      # initial version with label from tags
      # inner_results = [self.id_to_label[top_n_id], round(predictions.squeeze()[top_n_id].astype(float), 3)]

      # here we have multiple options:
      # add or not tags label
      # add all or one (first or random) qa label
      inner_results = []
      for label_qa in self.mappings[self.id_to_label[top_n_id]]:
        found = False
        for aux_result in res['results']:
          if label_qa == aux_result[0]:
            found = True
            break
        if found == False:
          inner_results.append([label_qa, round(predictions.squeeze()[top_n_id].astype(float), 3)])

      res['results'].extend(inner_results)

    # what to do when having less results?
    if len(res['results']) < n_hits:
      if n_hits > 200:
        res = {}
        res['results'] = [[self.id_to_label[top_n_index], round(predictions.squeeze()[top_n_index].astype(float), 3)] for top_n_index in top_n_idxs]
      else:
        res = self._post_process((predictions, 2*n_hits))
    
    # trim results to needed value
    res['results'] = res['results'][:n_hits]
    return res

if __name__ == '__main__':
  from libraries import Logger

  l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=True)

  a = GetQAWorker(log=l, default_config=_CONFIG, verbosity_level=0)
  a._load_model()
  ins = a._pre_process({"QUERY": "Cat la suta din salariul brut merge la pensii pentru un programator?", "TOP_N": 10})
  p = a._predict(ins)
  r = a._post_process(p)

  print(r)