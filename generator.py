# -*- coding: utf-8 -*-
"""
Copyright 2019-2021 Lummetry.AI (4E SOFTWARE SRL). All Rights Reserved.


* NOTICE:  All information contained herein is, and remains the property of 4E SOFTWARE SRL.  
* The intellectual and technical concepts contained herein are proprietary to 4E SOFTWARE SRL
* and may be covered by Romanian and Foreign Patents, patents in process, and are protected 
* by trade secret or copyright law.
* Dissemination of this information or reproduction of this material is strictly forbidden 
* unless prior written permission is obtained from 4E SOFTWARE SRL.
*
*
*  RO:
*    Modul software TempRent, proiect finanțat în cadrul POC, Axa prioritara 2 - Tehnologia Informației și Comunicațiilor (TIC) 
*    pentru o economie digitală competitivă, Prioritatea de investiții 2b - Dezvoltarea produselor și s
*    erviciilor TIC, a comerțului electronic și a cererii de TIC, cod SMIS 142474, 
*    Contractul de finanțare nr. 2/221_ap3/24.06.2021.
*

  
  RO:
    Acest produs a fost livrat si realizat in baza serviciilor de cercetare-inovare industrială 
    conform contract de servicii nr. 9 din 01.11.2021 folosind modulele AI "ALLAN" aferente "TempRent" -  
    Proiect finanțat în cadrul POC, Axa prioritara 2 - Tehnologia Informației și Comunicațiilor (TIC) 
    pentru o economie digitală competitivă, Prioritatea de investiții 2b - Dezvoltarea produselor și s
    erviciilor TIC, a comerțului electronic și a cererii de TIC, cod SMIS 142474, 
    Contractul de finanțare nr. 2/221_ap3/24.06.2021.
    
"""

import os
import time
import numpy as np

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

from libraries import Logger
from libraries import LummetryObject
from utils.utils import test_model
import constants as ct

__VER__ = '0.2.5.2'

class LossCallback(CallbackAny2Vec):
  '''Callback to print loss after each epoch.'''

  def __init__(self, log, max_epoch, model_no):
    self.epoch = 1
    self.log = log
    self.model_no = model_no
    self.max_epoch = max_epoch
    self.timings = []
    return
      
  def on_train_begin(self, model):
    self.log.P("Begin training model {} on with len(wv)={} {}".format(
      self.model_no,
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
      "Model {}, Epoch #{}, Loss: {}, Epoch time: {}, Elapsed time: {}, Remaining time: {}".format(
        self.model_no, self.epoch, loss,
        time.strftime("%H:%M:%S", time.gmtime(epoch_time)),
        time.strftime("%H:%M:%S", time.gmtime(elapsed_time)),
        time.strftime("%H:%M:%S", time.gmtime(remaining_time)),
        ),      
      color='g'
      )    
    model_fn = os.path.join(
      self.log.get_models_folder(),
      l.file_prefix + '_e{}_v{}K_ep_{:02}'.format(
        model.wv.vector_size, 
        len(model.wv) // 1000,
        self.epoch
        )
      )
    self.log.P("Saving '{}'".format(model_fn), color='g')
    model.save(model_fn)
    self.epoch += 1



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
    if len(files) == 0:
      raise ValueError("No data files to train on!")
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
    l.P("Detected running in debug mode.", color='y')
    params = [
        {'max_final_vocab': 120000, 'min_count':20 ,'vector_size':128, 'epochs':25, 'window':5},
      ]
    workers = 11
  else:
    max_vocab = None
    l.P("Detected running in live model", color='y')
    params = [
        {'max_final_vocab':   None, 'min_count':40, 'vector_size':128, 'epochs':40, 'window':5},
        {'max_final_vocab':   None, 'min_count':20, 'vector_size':128, 'epochs':40, 'window':5},
        {'max_final_vocab': 140000, 'min_count':20, 'vector_size':128, 'epochs':40, 'window':5},
        {'max_final_vocab': 120000, 'min_count':20, 'vector_size':128, 'epochs':40, 'window':5},
        
      ]
    data_folder = l.get_data_subfolder('_embeds_input')
    workers = 15

  for idx, param_set in enumerate(params):
    l.P("Training model {}/{} with {}".format(idx+1, len(params), param_set), color='g')
    cg = CorpusGenerator(
      log=l,
      )
    
    model = Word2Vec(
      sentences=cg,
      sg=1,
      workers=workers,    
      alpha=0.004,
      min_alpha=0.001,
      negative=20,
      compute_loss=True,
      callbacks=[LossCallback(log=l, max_epoch=param_set['epochs'], model_no=idx+1)],
      **param_set,
      )
    
    l.P("Test final:", color='g')
    test_model(
      log=l, 
      model=model,
      words=ct.WV.TEST_LIST,
      color='g',
      )
    
    model_fn = os.path.join(
      l.get_models_folder(),
      l.file_prefix + '_e{}_v{}K_final'.format(
        model.wv.vector_size, 
        len(model.wv) // 1000,
        )
      )
    model.save(model_fn)
    