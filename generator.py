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
@created on: Sat Nov 20 08:40:40 2021
@created by: damian


"""

import re
import os

import unicodedata
from gensim.models import Word2Vec
from gensim.test.utils import datapath
from gensim import utils
from gensim.models.callbacks import CallbackAny2Vec

from libraries.logger import Logger
from libraries.generic_obj import LummetryObject

import constants as ct

def strip_accents(s):
   return ''.join(c for c in unicodedata.normalize('NFD', s)
                  if unicodedata.category(c) != 'Mn')

class LossCallback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self, log):
        self.epoch = 0
        self.log = log

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.log.P('    Loss after epoch {}: {}'.format(self.epoch, loss))
        self.epoch += 1

class CorpusGenerator(LummetryObject):
  def __init__(self, folder, encoding=ct.WV.RO_ENCODING, **kwargs):
    self._folder = folder
    self._encoding = encoding
    self.tag_cleaner = re.compile('<.*?>')
    super().__init__(**kwargs)
    return
  
  
  def remove_exclusions(self, line):
    clean_line = re.sub(self.tag_cleaner, '', line)
    return clean_line
    
    
  def __iter__(self):
    self.P("Processing folder '{}'".format(self._folder))
    files = os.listdir(self._folder)
    for fn in files:
      full_path_fn = os.path.join(self._folder, fn)      
      self.P("  Processing file '{}'".format(full_path_fn))
      for line in open(full_path_fn, encoding=self._encoding):
        clean_line = self.remove_exclusions(line)
        preprocessed = utils.simple_preprocess(clean_line, deacc=True)
        if len(preprocessed) < 10:
          continue
        yield preprocessed
      
      
if __name__ == '__main__':
  
  l = Logger('LAI', base_folder='.', app_folder='_cache')
  model_fn = os.path.join(l.get_models_folder(), l.file_prefix + 'embeds')
  if l.is_running_from_ipython:
    data_folder = l.get_dropbox_subfolder('_allan_data/_indaco/_data')
    l.P("Detected running in debug mode. Using '{}'".format(data_folder))
  else:
    data_folder = l.get_data_subfolder('_embeds')
    l.P("Detected running in live model. Using '{}'".format(data_folder))
  
  cg = CorpusGenerator(data_folder, log=l)
  
  model = Word2Vec(
    sentences=cg,
    vector_size=200,
    min_count=10,
    sg=1,
    workers=7,    
    alpha=0.005,
    min_alpha=0.001,
    negative=20,
    epochs=25,
    compute_loss=True,
    callbacks=[LossCallback(log=l)],
    )
  
  model.save(model_fn)
  