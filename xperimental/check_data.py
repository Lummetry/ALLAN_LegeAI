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
@created on: Sat Dec  4 10:21:42 2021
@created by: damia
"""

import os
import numpy as np
import pandas as pd

from libraries import Logger

if __name__ == '__main__':
  fn_stats = '20211203_154548_x_data_dist.pkl'
  fn_data = '20211203_154548_x_data.pkl'
  l = Logger('DBTST', base_folder='.', app_folder='_cache', TF_KERAS=False)
  if not l.runs_from_ipython():
    data = l.load_pickle_from_data(fn_data)
  else:
    data = [['a']*int(np.random.normal(100,30)) for _ in range(1000)]
  
  margins = 10
  max_len = 1000
  min_len = 20
  lens = [len(x) for x in data]
  l.P("Raw lens: \n{}".format(pd.Series(lens).describe()))
  if not l.runs_from_ipython():
    l.save_pickle_to_data(lens, fn_stats)
  else:
    fld = l.get_dropbox_subfolder('_allan_data/_indaco/_data/tagger_dataset')
    fn = os.path.join(fld, fn_stats)
    if l.get_data_file(fn):
      lens = l.load_pickle_from_data(fn)
  lens = [x for x in lens if x < max_len and x > min_len]
  bins = np.bincount(lens)
  max_pos = np.argmax(bins)
  left = max_pos - margins
  right = max_pos + margins
  total = 0
  for i in range(left, right +1):
    l.P('Word: {} => {} docs ({:.1f}%)'.format(i, bins[i], bins[i]/len(lens)*100))
    total += bins[i]
  l.P("Total {} ({:.1f}%) of docs in range {} to {} words".format(
    total, total/len(lens)*100, left, right))
  l.P("Filtered lens:\n{}".format(pd.Series(lens).describe()))
  if l.runs_from_ipython():
    import matplotlib.pyplot as plt
    plt.hist(lens, bins=200)
    plt.xlabel('words per doc')
    plt.ylabel('freq of docs')
    plt.show()
    