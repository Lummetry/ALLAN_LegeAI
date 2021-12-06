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

import tensorflow as tf
import random
import os


from functools import partial

def test_tf_ds(dataset):
  for x,y in iter(dataset):
    print("{}, {}".format(x.shape, y.shape))
    print(x)
    print(y)
  return

def get_generator(corpus, lbls, emb_approximator, fixed_length=50, convert_unk_word=False):
  while True:
    for i in range(len(corpus)):
      if convert_unk_word is False:
        X, y = emb_approximator.encode(
          text=[corpus[i]],
          direct_embeddings=True,
          text_label=[lbls[i]],
          fixed_len=fixed_length,
          raw_conversion=True,
          convert_unknown_words=False,
          DEBUG=False
        )
      else:
        X, y = emb_approximator.encode(
          text=[corpus[i]],
          direct_embeddings=True,
          text_label=[lbls[i]],
          fixed_len=fixed_length,
          raw_conversion=True,
          convert_unknown_words=True,
          DEBUG=False
        )
      yield X.squeeze(axis=0), y.squeeze(axis=0)

def dataset(log, lst_X_paths, lst_y_paths, batch_size, emb_approximator, subfolder_path=None, fixed_length=50):
  corpus = []
  lbls = []

  for x_path, y_path in zip(lst_X_paths, lst_y_paths):
    corpus += log.load_pickle_from_data(x_path, subfolder_path=subfolder_path)
    lbls += log.load_pickle_from_data(y_path, subfolder_path=subfolder_path)

  c = list(zip(corpus, lbls))
  random.shuffle(c)

  dataset_size = len(c)

  train_br = int(0.7 * dataset_size)
  dev_br = int(0.85 * dataset_size)
  test_br = dataset_size

  train_corpus, train_lbls = zip(*c[:train_br])
  dev_corpus, dev_lbls = zip(*c[train_br:dev_br])
  test_corpus, test_lbls = zip(*c[dev_br:test_br])

  output_types = (tf.float32, tf.int32)
  output_shapes = ((fixed_length, emb_approximator.emb_size), (emb_approximator.output_size))

  #### Train:
  train_generator = partial(
    get_generator,
    corpus=train_corpus, lbls=train_lbls,
    emb_approximator=emb_approximator,
    fixed_length=fixed_length,
  )
  train_steps_per_epoch = train_br // batch_size + 1

  train_dataset = tf.data.Dataset.from_generator(
    train_generator,
    output_types=output_types,
    output_shapes=output_shapes
  ).batch(batch_size).prefetch(3)

  #### Dev:
  dev_generator = partial(
    get_generator,
    corpus=dev_corpus, lbls=dev_lbls,
    emb_approximator=emb_approximator,
    fixed_length=fixed_length,
  )
  dev_steps_per_epoch = (dev_br - train_br) // batch_size + 1

  dev_dataset = tf.data.Dataset.from_generator(
    dev_generator,
    output_types=output_types,
    output_shapes=output_shapes
  ).batch(batch_size).prefetch(3)

  #### Test:
  test_generator = partial(
    get_generator,
    corpus=test_corpus, lbls=test_lbls,
    emb_approximator=emb_approximator,
    fixed_length=fixed_length,
    convert_unk_word=True
  )
  test_steps_per_epoch = (test_br - dev_br) // batch_size + 1

  test_dataset = tf.data.Dataset.from_generator(
    test_generator,
    output_types=output_types,
    output_shapes=output_shapes
  ).batch(batch_size).prefetch(3)


  return train_dataset, train_steps_per_epoch, dev_dataset, dev_steps_per_epoch, test_dataset, test_steps_per_epoch


if __name__ == '__main__':
  from libraries import Logger
  from tagger.brain.emb_aproximator import EmbeddingApproximator
  EMB_MODEL_NAME = 'test_model'
  DATA_SUBFOLDER_PATH = 'tagger_dataset'
  DATA_MAPPER_FN = '{}/data_mapper.json'.format(DATA_SUBFOLDER_PATH)
  DCT_LBL_FN = '{}/dict_lbl.pk'.format(DATA_SUBFOLDER_PATH)

  log = Logger(lib_name='test_dataset', config_file='tagger/brain/configs/20211202/config_test_data.txt')

  dct_data_mapper = log.load_data_json(DATA_MAPPER_FN)
  dct_lbls = log.load_pickle_from_data(DCT_LBL_FN)
  emb_approximator = EmbeddingApproximator(
    log=log,
    dict_label2index=dct_lbls
  )

  tf_ds, steps_per_epoch = dataset(
    log=log,
    lst_X_paths=dct_data_mapper['train']['X'],
    lst_y_paths=dct_data_mapper['train']['y'],
    batch_size=2,
    emb_approximator=emb_approximator,
    subfolder_path=DATA_SUBFOLDER_PATH,
    fixed_length=50
  )

  test_tf_ds(tf_ds)

