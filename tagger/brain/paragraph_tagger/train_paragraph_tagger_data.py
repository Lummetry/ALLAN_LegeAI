"""
Copyright 2019 Lummetry.AI (Knowledge Investment Group SRL). All Rights Reserved.


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

"""

import os
import cv2
import numpy as np
from tqdm import tqdm
import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


from functools import partial

def test_tf_ds(dataset):
  for x,y in iter(dataset):
    print("{}, {}".format(x.shape, y.shape))
    print(x)
    print(y)
  return

def get_generator(corpus, lbls, emb_approximator, fixed_length=50):
  while True:
    for i in range(len(corpus)):
      with tf.device('/cpu:0'):
        X, y = emb_approximator.encode(
          text=[corpus[i]],
          direct_embeddings=True,
          text_label=[lbls[i]],
          fixed_len=50,
          raw_conversion=True,
          to_onehot=True,
          convert_unknown_words=False
        )
      yield X.squeeze(axis=0), y.squeeze(axis=0)

# def encode(X, y, emb_approximator, fixed_length=50):
#   with tf.device('/cpu:0'):
#
#   return X, y


def dataset(log, data_file, batch_size, emb_approximator, fixed_length=50, emb_size=128):
  dct_data = log.load_data_json(data_file)
  corpus = dct_data['X']
  lbls = dct_data['y']
  #### PREPARED PATHS:
  generator = partial(
    get_generator,
    corpus=corpus, lbls=lbls,
    emb_approximator=emb_approximator,
    fixed_length=fixed_length,
  )
  steps_per_epoch = len(corpus) // batch_size + 1

  output_types = (tf.float32, tf.int32)
  output_shapes = ((fixed_length, emb_size), (emb_approximator.output_size))

  dataset = tf.data.Dataset.from_generator(
    generator,
    output_types=output_types,
    output_shapes=output_shapes
  )


  dataset = dataset.batch(batch_size).prefetch(1)


  return dataset, steps_per_epoch


if __name__ == '__main__':
  from libraries import Logger
  from tagger.brain.emb_aproximator import EmbeddingApproximator
  EMB_MODEL_NAME = 'test_model'
  DATA_FN = 'test_data.json'
  DCT_LBL_FN = 'dict_lbl.pk'

  log = Logger(lib_name='test_dataset', config_file='tagger/brain/configs/20211202/config_test_data.txt')

  dct_lbls = log.load_pickle_from_data(DCT_LBL_FN)
  with tf.device('/cpu:0'):
    emb_approximator = EmbeddingApproximator(
      log=log,
      dict_label2index=dct_lbls
    )
    # emb_approximator.maybe_load_pretrained_embgen(embgen_model_file=EMB_MODEL_NAME)
    # emb_approximator._get_generated_embeddings()

    tf_ds, steps_per_epoch = dataset(
      log=log,
      data_file=DATA_FN,
      batch_size=2,
      emb_approximator=emb_approximator,
      fixed_length=50
    )
    x, y = emb_approximator.encode(
      text="ANA ARE MERE",
      direct_embeddings=True,
      text_label=[["L1", "L2", "L3"]],
      fixed_len=50,
      raw_conversion=True,
      to_onehot=True,
      convert_unknown_words=False
    )
    test_tf_ds(tf_ds)

