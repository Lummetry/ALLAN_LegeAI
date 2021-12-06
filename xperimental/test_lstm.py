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
@created on: Thu Nov 25 14:31:39 2021
@created by: razvan
"""
import numpy as np
import tensorflow as tf


if __name__ == '__main__':
  inp = tf.keras.layers.Input((10,1))
  x = tf.keras.layers.LSTM(32)(inp)
  mdl = tf.keras.models.Model(inp, x)

  input = np.arange(10).reshape((1,10,1))

  res = mdl(input)
  print(res)