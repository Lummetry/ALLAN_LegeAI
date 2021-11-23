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

from gensim.models import Word2Vec

from libraries.logger import Logger

from embeds_utils.utils import test_model

if __name__ == '__main__':
  
  FN1 = '20211123_112106embeds_ep_04'
  FN2 = '20211123_112106embeds_ep_11'
  
  l = Logger('LAIEMB', base_folder='.', app_folder='_cache')
  fn1 = l.get_models_file(FN1)
  fn2 = l.get_models_file(FN2)
  l.P("Loading '{}'...".format(fn1))
  m1 = Word2Vec.load(fn1)
  l.P("Loading '{}'...".format(fn2))
  m2 = Word2Vec.load(fn2)
  
  test_model(log=l, model=m1, name='Epoch 4')
  
  test_model(log=l, model=m2, name='Epoch 11')
  