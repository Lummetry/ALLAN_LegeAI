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
  def __init__(self, folder, **kwargs):
    self._folder = folder
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
      self.P("  Processing file '{}'".format(fn))
      path = os.path.join(self._folder, fn)      
      corpus_path = datapath(path)
      for line in open(corpus_path, encoding="latin-1"):
        clean_line = self.remove_exclusions(line)
        preprocessed = utils.simple_preprocess(clean_line)
        if len(preprocessed) < 10:
          continue
        yield preprocessed
      
      
if __name__ == '__main__':
  l = Logger('LAI', base_folder='Dropbox', app_folder='_allan_data/_indaco')
  model_fn = os.path.join(l.get_models_folder(), l.file_prefix + 'embeds')
  
  cg = CorpusGenerator(l.get_data_folder(), log=l)
  
  model = Word2Vec(
    sentences=cg,
    vector_size=200,
    workers=7,
    compute_loss=True,
    callbacks=[LossCallback(log=l)],
    )
  
  model.save(model_fn)
  