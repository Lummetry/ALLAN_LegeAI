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
@created by: damia
"""
import numpy as np

from gensim.models import Word2Vec

from libraries.logger import Logger

if __name__ == '__main__':
  l = Logger('LAICNV', base_folder='.', app_folder='_cache')
  SPECIALS = [
    "<PAD>",
    "<UNK>",
    "<SOS>",
    "<EOS>",
  ]
  n_special = len(SPECIALS)
  FN = '20211124_082955_e128_v191K_final'
  model_fn = l.get_models_file(FN)
  model = Word2Vec.load(model_fn)
  if model is not None:
    np_gemb = model.wv.vectors
    np_spec = np.ones((n_special, np_gemb.shape[1])) * np.array([x * 0.0001 for x in range(n_special)]).reshape((-1,1))
    np_emb = np.concatenate(
      (
        np_spec,
        np_gemb
        ),
      axis=0,
      dtype='float32'
      )
    lst_keys = SPECIALS + model.wv.index_to_key
    dct_i2w = {i:k for i, k in enumerate(lst_keys)}
    l.save_pickle_to_data(np_emb, 'lai_embeddings_{}K.pkl'.format(len(dct_i2w)//1000))
    l.save_pickle_to_data(dct_i2w, 'lai_ro_i2w_{}K.pkl'.format(len(dct_i2w)//1000))
    
  