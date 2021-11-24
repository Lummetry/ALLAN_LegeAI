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
from embeds_utils.utils import test_model
import constants as ct

__VER__ = '0.2.0.0'

class LossCallback(CallbackAny2Vec):
  '''Callback to print loss after each epoch.'''

  def __init__(self, log, model_fn, max_epoch):
    self.epoch = 1
    self.log = log
    self.model_fn = model_fn
    self.max_epoch = max_epoch
    self.timings = []
    return
      
  def on_train_begin(self, model):
    self.log.P("Begin training on with len(wv)={} {}".format(
      len(model.wv), ' ' * 20,
      ), color='g')
    return
      
  def on_epoch_begin(self, model):
    self.start_time = time.time()

  def on_epoch_end(self, model):
    loss = model.get_latest_training_loss()
    epoch_time = time.time() - self.start_time
    self.timings.append(epoch_time)
    elapsed_time = np.sum(self.timings)
    remaining_time = (self.max_epoch - self.epoch) * np.mean(self.timings)
    test_model(
      log=self.log,
      model=model,
      name="Epoch {}".format(self.epoch)
      )
    self.log.P(
      "Epoch #{}, Loss: {}, Epoch time: {}, Elapsed time: {}, Remaining time: {}".format(
        self.epoch, loss,
        time.strftime("%H:%M:%S", time.gmtime(epoch_time)),
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
    self.version = __VER__
    super().__init__(**kwargs)
    return
  
  def __iter__(self):
    folder = self.log.get_data_folder()
    self.P("Processing folder '{}'".format(folder))
    files = [x for x in os.listdir(folder) if self.file_prefix in x and '.pkl' in x]
    for fn in files:
      self.P("  Processing file '{}' {}".format(fn, ' ' * 50))
      wordlist = l.load_pickle_from_data(fn)
      nr_batches = len(wordlist) // self.batch_size
      step = nr_batches // 100
      for batch_idx in range(nr_batches):
        start = batch_idx * self.batch_size
        end = (batch_idx + 1) * self.batch_size
        words =  wordlist[start:end]
        if batch_idx % step == 0:
          print("\rProcessing '{}': {:.1f}% - {} {}\r".format(
            fn,
            (batch_idx + 1) / nr_batches * 100,
            words[:4],
            ' ' * 30,
            ), end='', flush=True)
        yield words
      
      
if __name__ == '__main__':
  FORCE_LOCAL = False
  l = Logger('LAI', base_folder='.', app_folder='_cache')
  
  if l.is_running_from_ipython and not FORCE_LOCAL:
    max_vocab = 150000
    l.P("Detected running in debug mode. Using 'small' vocab size {}".format(
      max_vocab), color='y')
    epochs = 25
    workers = 11
    emb_size=128
    window = 5
    min_count = 20
  else:
    max_vocab = None
    l.P("Detected running in live model. Using vocab size {}".format(
      max_vocab), color='y')
    data_folder = l.get_data_subfolder('_embeds_input')
    emb_size=128
    epochs = 40
    workers = 15
    window = 5
    min_count = 40

  model_fn = os.path.join(l.get_models_folder(), l.file_prefix + 'emb{}'.format(emb_size))
  
  cg = CorpusGenerator(
    log=l,
    )
  
  model = Word2Vec(
    sentences=cg,
    vector_size=emb_size,
    window=window,
    min_count=min_count,
    sg=1,
    workers=workers,    
    alpha=0.004,
    min_alpha=0.001,
    negative=20,
    epochs=epochs,
    compute_loss=True,
    max_final_vocab=max_vocab,
    callbacks=[LossCallback(log=l, model_fn=model_fn, max_epoch=epochs)],
    )
  
  l.P("Test final:", color='g')
  test_model(
    log=l, 
    model=model,
    words=ct.WV.TEST_LIST,
    color='g',
    )
  model.save(model_fn)
  