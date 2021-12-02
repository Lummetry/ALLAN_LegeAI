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

from libraries import Logger

import unicodedata

import constants as ct

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')
 
  
if __name__ == '__main__':
  URL = 'https://www.dropbox.com/s/lfeupp18fp5efjy/20200120_corpus_merged?dl=1'
  l = Logger('LAIT1', base_folder='.', app_folder='_cache')
  fn_in = l.maybe_download(url=URL, fn='rowiki_raw_unicode', target='data')
  fn_out = os.path.join(l.get_data_folder(), 'corpus_merged_normalized')
  if os.path.isfile(fn_out):
    l.P("'{}' already exists".format(fn_out))
  else:
    fh_in = open(fn_in, encoding='utf-8')
    fh_out = open(fn_out, 'wt', encoding=ct.WV.RO_ENCODING)
    for line in fh_in:
      line_out = strip_accents(line)
      b_line = line_out.encode(ct.WV.RO_ENCODING, 'ignore')
      line_out_converted = b_line.decode(ct.WV.RO_ENCODING, 'ignore')
      fh_out.write(line_out_converted)
    fh_out.close()
    fh_in.close()
    