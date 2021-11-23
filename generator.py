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

import os
import time
import numpy as np

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from libraries.logger import Logger
from libraries.generic_obj import LummetryObject

EPOCHS = 15

class LossCallback(CallbackAny2Vec):
  '''Callback to print loss after each epoch.'''

  def __init__(self, log, model_fn):
    self.epoch = 1
    self.log = log
    self.model_fn = model_fn
    self.timings = []
    return
      
  def on_train_begin(self, model):
    self.log.P("Begin training on with len(wv)={}".format(len(model.wv)), color='g')
    return
      
  def on_epoch_begin(self, model):
    self.start_time = time.time()

  def on_epoch_end(self, model):
    loss = model.get_latest_training_loss()
    elapsed_time = time.time() - self.start_time
    remaining_time = (EPOCHS - self.epoch) * np.mean(elapsed_time)
    self.timings.append(elapsed_time)
    self.log.P(
      "Epoch #{} end - Loss: {}, Elapsed time: {}, Remaining time: {}".format(
        self.epoch, loss,
        time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
        time.strftime("%H:%M:%S", time.gmtime(remaining_time)),
        ),      
      color='g'
      )    
    self.epoch += 1
    model_fn = self.model_fn + '_ep_{:02}'.format(self.epoch)
    self.log.P("Saving '{}'".format(model_fn), color='g')
    model.save(model_fn)



class CorpusGenerator(LummetryObject):
  def __init__(self, file_prefix='preproc', batch_size=1000, **kwargs):
    self.file_prefix = file_prefix
    self.batch_size = batch_size
    super().__init__(**kwargs)
    return
  
  def __iter__(self):
    folder = self.log.get_data_folder()
    self.P("Processing folder '{}'".format(folder))
    files = [x for x in os.listdir(folder) if self.file_prefix in x and '.pkl' in x]
    for fn in files:
      self.P("  Processing file '{}'".format(fn))
      wordlist = l.load_pickle_from_data(fn)
      nr_batches = len(wordlist) // self.batch_size
      step = nr_batches // 100
      for batch_idx in range(nr_batches):
        start = batch_idx * self.batch_size
        end = (batch_idx + 1) * self.batch_size
        if batch_idx % step == 0:
          print("\rProcessing '{}': {:.1f}%\r".format(
            fn,
            (batch_idx + 1) / nr_batches * 100.
            ), end='', flush=True)
        yield wordlist[start:end]
      
      
if __name__ == '__main__':
  os.system('color')
  FORCE_LOCAL = False
  l = Logger('LAI', base_folder='.', app_folder='_cache')
  model_fn = os.path.join(l.get_models_folder(), l.file_prefix + 'embeds')
  if l.is_running_from_ipython and not FORCE_LOCAL:
    max_vocab = 150000
    l.P("Detected running in debug mode. Using 'small' vocab size {}".format(
      max_vocab), color='y')
  else:
    data_folder = l.get_data_subfolder('_embeds_input')
    max_vocab = None
    l.P("Detected running in live model. Using vocab size {}".format(
      max_vocab), color='y')
  
  cg = CorpusGenerator(
    log=l,
    )
  
  model = Word2Vec(
    sentences=cg,
    vector_size=128,
    window=5,
    min_count=15,
    sg=1,
    workers=10,    
    alpha=0.005,
    min_alpha=0.001,
    negative=20,
    epochs=EPOCHS,
    compute_loss=True,
    max_final_vocab=max_vocab,
    callbacks=[LossCallback(log=l, model_fn=model_fn)],
    )
  
  model.save(model_fn)
  