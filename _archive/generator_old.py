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

from gensim.models import Word2Vec
from gensim import utils
from gensim.models.callbacks import CallbackAny2Vec

from libraries import Logger
from libraries import LummetryObject

import constants as ct


class LossCallback(CallbackAny2Vec):
  '''Callback to print loss after each epoch.'''

  def __init__(self, log, model_fn):
    self.epoch = 1
    self.log = log
    self.model_fn = model_fn
    return
      
  def on_train_begin(self, model):
    self.log.P("Begin training on with len(wv)={}".format(len(model.wv)))
    return
      

  def on_epoch_end(self, model):
    loss = model.get_latest_training_loss()
    self.log.P('Loss after epoch {}: {}'.format(self.epoch, loss), color='g')
    self.epoch += 1
    model.save(self.model_fn + '_ep_{:02}'.format(self.epoch))



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
      if not os.path.isfile(full_path_fn):
        continue
      self.P("  Processing file '{}'".format(full_path_fn))
      for line in open(full_path_fn, encoding=self._encoding):
        clean_line = self.remove_exclusions(line)
        preprocessed = utils.simple_preprocess(clean_line, deacc=True)
        if len(preprocessed) < 10:
          continue
        yield preprocessed
      
      
if __name__ == '__main__':
  FORCE_LOCAL = False
  l = Logger('LAI', base_folder='.', app_folder='_cache')
  model_fn = os.path.join(l.get_models_folder(), l.file_prefix + 'embeds')
  if l.is_running_from_ipython and not FORCE_LOCAL:
    data_folder = l.get_dropbox_subfolder('_allan_data/_indaco/_data')
    l.P("Detected running in debug mode. Using '{}'".format(data_folder))
  else:
    data_folder = l.get_data_subfolder('_embeds_input')
    l.P("Detected running in live model. Using '{}'".format(data_folder))
  
  cg = CorpusGenerator(data_folder, log=l)
  
  model = Word2Vec(
    sentences=cg,
    vector_size=200,
    min_count=10,
    sg=1,
    workers=10,    
    alpha=0.005,
    min_alpha=0.001,
    negative=20,
    epochs=15,
    compute_loss=True,
    callbacks=[LossCallback(log=l, model_fn=model_fn)],
    )
  
  model.save(model_fn)
  