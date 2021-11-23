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
               words=['leafa', 'tva', 'bacsis', 'munca'], 
               name=None,
               topn=5,
               ):
  if name is not None:
    log.P("Test for model '{}'".format(name), color='b')
  for _w in words:
    res = model.wv.most_similar(_w, topn=topn)
    log.P("  Top {} for '{}': {}".format(
      topn, _w,
      ", ".join(["{}: {:.2f}".format(_sw, _sc) for _sw, _sc in res ])
      ), color='b')
  return