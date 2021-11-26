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
@author: Lummetry.AI - AID
@project:
@description:

"""

def test_model(log, 
               model, 
               words=['leafa', 'salarizare', 'impozit', 'tva', 'divort', 'munca', 'frauda', 'copil'], 
               name=None,
               topn=5,
               color='b',
               ):
  if name is not None:
    log.P("Test for model '{}' {}".format(name, ' ' * 20), color=color)
  for _w in words:
    res = model.wv.most_similar(_w, topn=topn)
    log.P("  Top {} for '{}': {}".format(
      topn, _w,
      ", ".join(["{}: {:.2f}".format(_sw, _sc) for _sw, _sc in res ])
      ), color=color)
  if name is not None:
    log.P("*" * 80)
  return


def K_triplet_loss(y_pred, beta=0.5,):
   import tensorflow.keras.backend as K

   anchor = y_pred[0]
   positive = y_pred[1]
   negative = y_pred[2]

   similar_dist = K.sum(K.square(anchor - positive), axis=1)
   diff_dist = K.sum(K.square(anchor - negative), axis=1)
   loss = K.maximum(similar_dist - diff_dist + beta, 0.0)
   loss = K.expand_dims(loss)
   return loss

def K_identity_loss(y_true, y_pred):
  import tensorflow.keras.backend as K
  return K.mean(y_pred - 0 * y_true)