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
import os
import re
from gensim import utils

from libraries.logger import Logger

import constants as ct


if __name__ == '__main__':
  l = Logger('LAIP', base_folder='.', app_folder='_cache')
  WORDS_PER_FILE = 50000000
  folder = l.get_dropbox_subfolder('_allan_data/_indaco/_data')
  files = os.listdir(folder)
  wordlist = []
  file_no = 1
  fn_out_base = 'preproc_{:02}.pkl'
  fn_out = fn_out_base.format(file_no)
  tag_cleaner = re.compile('<.*?>')
  l.P("Preprocessing folder '{}'".format(folder))
  for _fn in files:
    fn = os.path.join(folder, _fn)
    if not os.path.isfile(fn):
      continue
    l.P("  Preprocessing file '{}'".format(fn))
    with open(fn, encoding=ct.WV.RO_ENCODING) as fh:
      for line in fh:
        clean_line = re.sub(tag_cleaner, '', line)
        preprocessed = utils.simple_preprocess(clean_line, deacc=True)
        wordlist += preprocessed
        if (len(wordlist) % 1000) == 0:
          print("\rPreparing file '{}': {:.1f}%\r".format(
            fn_out, len(wordlist) / WORDS_PER_FILE * 100), end='', flush=True)
        if len(wordlist) >= WORDS_PER_FILE:
          l.save_pickle_to_data(wordlist, fn_out)
          wordlist = []
          file_no += 1
          fn_out = fn_out_base.format(file_no)
  if len(wordlist) > 0:
    l.save_pickle_to_data(wordlist, fn_out)
    
          
      
    
    