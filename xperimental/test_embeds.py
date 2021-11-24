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
import constants as ct

if __name__ == '__main__':
  
  fn_models = [
    '20211123_112106embeds_ep_04',
    '20211123_112106embeds_ep_11',  
    '20211123_183830embeds',
    ]
  
  l = Logger('LAIEMB', base_folder='.', app_folder='_cache')
  fns = []
  models = {}
  for fn in fn_models:
    fns.append(l.get_models_file(fn))
    l.P("Loading '{}'...".format(fns[-1]))
    m = Word2Vec.load(fns[-1])
    models[fn] = m
    
  for name, model in models.items():
    test_model(
      log=l, 
      model=model, name=name,
      words=ct.WV.TEST_LIST,
      )

  