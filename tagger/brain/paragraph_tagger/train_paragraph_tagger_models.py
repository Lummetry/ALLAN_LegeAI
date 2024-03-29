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
import numpy as np

from libraries import Logger



def get_tnn_column(log, tf_x,  lst_filters, kernel_size, end, strides, causal=False, act='relu',
                   base_name='col'):

  if causal:
    padding = 'causal'
  else:
    padding = 'valid'
  #endif

  for i in range(len(lst_filters)):
    lyr = tf.keras.layers.Conv1D(
      filters=lst_filters[i],
      kernel_size=kernel_size,
      strides=strides,
      padding=padding,
      name='{}_d{}_size{}_filters{}'.format(base_name, i, kernel_size, lst_filters[i]),
      activation=None
    )
    tf_x = lyr(tf_x)
    tf_x = tf.keras.layers.BatchNormalization(name='{}_d{}_bn'.format(base_name, i))(tf_x)
    tf_x = tf.keras.layers.Activation(act, name='{}_d{}_act_{}'.format(base_name, i, act))(tf_x)
  # endfor

  if end == 'gp':
    tf_x1 = tf.keras.layers.GlobalMaxPool1D(name='{}_max_pool'.format(base_name))(tf_x)
    tf_x2 = tf.keras.layers.GlobalAvgPool1D(name='{}_avg_pool'.format(base_name))(tf_x)
    tf_x = tf.keras.layers.concatenate([tf_x1, tf_x2], name='{}_concat_gp'.format(base_name))
  elif end == 'lstm':
    lstm_lyr = tf.keras.layers.LSTM(lst_filters[-1], name='{}_lstm'.format(base_name))
    bi_lyr = tf.keras.layers.Bidirectional(lstm_lyr, name='{}_bidirectional'.format(base_name))
    tf_x = bi_lyr(tf_x)

  return tf_x


def get_model(log, input_shape, nr_outputs, optimizer, **kwargs):
  UNITS_DENSE = kwargs.get('UNITS_DENSE')
  MODEL_NAME = kwargs.get('MODEL_NAME')
  DROP_PRE_READOUT = kwargs.get('DROP_PRE_READOUT', 0)
  FILTERS = kwargs.get("FILTERS")
  KERNELS = kwargs.get("KERNELS")
  STRIDES = kwargs.get("STRIDES")
  ENDS = kwargs.get("ENDS")
  CNN_ACT = kwargs.get("CNN_ACT")
  ACTIV_DENSE = kwargs.get("ACTIV_DENSE")

  assert len(FILTERS) == len(KERNELS)

  UNITS_READOUT = nr_outputs
  OPTIMIZER = optimizer

  tf_inp = tf.keras.layers.Input(shape=input_shape, name='inp')
  tf_x = tf_inp

  columns = []
  for i in range(len(FILTERS)):
    tf_x_column = get_tnn_column(
      log=log,
      tf_x=tf_x,
      lst_filters=FILTERS[i],
      kernel_size=KERNELS[i],
      strides=STRIDES[i],
      end=ENDS[i],
      base_name='col_{}'.format(i),
      act=CNN_ACT,
    )
    columns.append(tf_x_column)
  if len(columns) > 1:
    tf_x = tf.keras.layers.concatenate(columns, name='concat')
  else:
    tf_x = columns[0]

  for i, units in enumerate(UNITS_DENSE):
    tf_x = tf.keras.layers.Dense(
      units=units,
      name='pre_readout_dense_{}'.format(i)
    )(tf_x)
    tf_x = tf.keras.layers.BatchNormalization(name='pre_readout_bn{}'.format(i))(tf_x)
    tf_x = tf.keras.layers.Activation(ACTIV_DENSE, name='pre_readout_act_{}_{}'.format(ACTIV_DENSE, i))(tf_x)
    if DROP_PRE_READOUT > 0:
      tf_x = tf.keras.layers.Dropout(rate=DROP_PRE_READOUT, name='drop_{}_{}'.format(DROP_PRE_READOUT, i))(tf_x)
    # endif
  # endfor

  act = 'sigmoid'
  loss = 'binary_crossentropy' #if UNITS_READOUT == 1 else 'categorical_crossentropy'

  tf_x = tf.keras.layers.Dense(
    units=UNITS_READOUT,
    activation=act,
    name='readout_{}'.format(act)
  )(tf_x)
  tf_out = tf_x

  model = tf.keras.models.Model(
    inputs=tf_inp,
    outputs=tf_out,
    name=MODEL_NAME
  )
  model.compile(
    loss=loss,
    optimizer=OPTIMIZER,
    metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
  )
  log.log_keras_model(model)
  return model


if __name__ == '__main__':
  log = Logger(
    lib_name='test_model',
    config_file='tagger/brain/configs/20211202/config_test_models.txt',
    TF_KERAS=True
  )

  SENTANCE_LENGTH = 500
  EMB_SIZE = 64

  dct_lbls = {
    'l0':0,
    'l1':1,
    'l2':2
  }

  model = get_model(
    log,
    input_shape=(500,64),
    nr_outputs=len(dct_lbls),
    optimizer='adam',
    MODEL_NAME='test_model',
    UNITS_DENSE=[128,32],
    DROP_PRE_READOUT=0.2,
    FILTERS=[
      [32,64,256,511],
      [1024,1024],
      [2,4,8,32,64,128]
    ],
    KERNELS=[3, 3, 3],
    CNN_ACT='relu'

  )

