"""
Copyright 2019 Lummetry.AI (Knowledge Investment Group SRL). All Rights Reserved.


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

"""


import numpy as np
import tensorflow as tf
import os
# from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from tqdm import tqdm

from libraries import Logger
from tagger.brain.emb_aproximator import EmbeddingApproximator
from tagger.brain.paragraph_tagger.train_paragraph_tagger_data import dataset
from tagger.brain.paragraph_tagger.train_paragraph_tagger_models import get_model

def multiclass_acc(y, y_hat):
  m = tf.keras.metrics.Accuracy()
  m.update_state(y, y_hat)
  return m.result().numpy()

def multiclass_rec(y, y_hat):
  m = tf.keras.metrics.Recall()
  m.update_state(y, y_hat)
  return m.result().numpy()


def evaluate_callback(model, dev_gen, steps_per_epoch, key, thrs=0.5):
  log.p("Evaluation for: {}".format(key))
  lst_y, lst_y_hat = [], []
  dev_iter = iter(dev_gen)
  for i in tqdm(range(steps_per_epoch)):
    x_batch, y_batch = next(dev_iter)
    lst_y_hat.append(model.predict(x_batch))
    lst_y.append(y_batch)

  y = np.vstack(lst_y)
  y_hat = np.vstack(lst_y_hat)
  y_pred = (y_hat > thrs).astype(np.uint8)

  acc = multiclass_acc(y, y_pred)
  rec = multiclass_rec(y, y_pred)
  return {'{}_acc'.format(key.lower()) : acc, '{}_rec'.format(key.lower()): rec}

def save_model_callback(log, model, s_name, delete_prev_named=False, DEBUG=False):

  debug = (not delete_prev_named) or DEBUG

  if debug:
    log.P("Saving tagger model '{}'".format(s_name))
  fn = os.path.join(log.get_models_folder(), log.file_prefix + '_' + s_name +'.h5')
  model.save(fn)

  return

def train_loop(log, train_dataset, dev_dataset, batch_size, n_epochs, steps_per_epoch,  model,
               eval_callback,
               save_model_callback,
                save_best=True,
                save_end=True,
                test_every_epochs=1,
                DEBUG=True,
                compute_topic=True):
  """
  this is the basic 'protected' training loop loop that uses tf.keras methods and
  works both on embeddings inputs or tokenized inputs
  """

  train_losses = []
  best_recall = 0
  train_recall_history = []
  train_acc_history = []
  # train_recall_history_epochs = []
  train_epoch = 0

  for epoch in range(n_epochs):
    train_epoch = epoch + 1
    hist = model.fit(
      train_dataset,
      epochs=1,
      verbose=1,
      steps_per_epoch=steps_per_epoch
    )
    dct_hist = {k: v[0] for i, (k, v) in enumerate(hist.history.items()) }
    train_losses.append(dct_hist['loss'])
    train_recall_history.append(dct_hist['recall'])
    train_acc_history.append( dct_hist['accuracy'])

    # if (epoch > 0) and (test_every_epochs > 0) and (dev_dataset is not None) and (
    #         (epoch + 1) % test_every_epochs == 0):
    log.P("Testing on epoch {}".format(epoch + 1))
    dct_eval = eval_callback(
      model=model,
      dev_gen=dev_dataset,
      steps_per_epoch=steps_per_epoch,
      key='dev'
    )
    rec = dct_eval['dev_rec']
    acc = dct_eval['dev_acc']

    if best_recall < rec:
      s_name = 'ep{}_R{:.2f}_A{:.2f}'.format(epoch + 1, rec, acc)
      save_model_callback(
        log=log,
        model=model,
        s_name=s_name,
        delete_prev_named=False
      ) ### TODO: implement delete prev
      best_recall = rec

  log.P("Model training done.")
  log.P("Train recall history: {}".format(train_recall_history))


  return train_recall_history

if __name__ == '__main__':
  EMB_MODEL_NAME = 'test_model'
  DATA_SUBFOLDER_PATH = 'tagger_dataset'
  DATA_MAPPER_FN = '{}/data_mapper.json'.format(DATA_SUBFOLDER_PATH)
  DCT_LBL_FN = '{}/dict_lbl.pk'.format(DATA_SUBFOLDER_PATH)

  LOGGER_CONFIG = 'tagger/brain/configs/20211202/config_train.txt'
  MODELS_DEF_FN = 'tagger/brain/configs/20211202/models_defs.json'

  FIXED_LENGTH = 50
  BATCH_SIZE = 512
  NR_EPOCHS = 2

  log = Logger(lib_name='TRN', config_file=LOGGER_CONFIG)
  model_defs = log.load_json(MODELS_DEF_FN)
  dct_data_mapper = log.load_data_json(DATA_MAPPER_FN)

  dct_lbls = log.load_pickle_from_data(DCT_LBL_FN)

  emb_approximator = EmbeddingApproximator(
    log=log,
    dict_label2index=dct_lbls
  )

  train_dataset, train_steps_per_epoch = dataset(
    log=log,
    lst_X_paths=dct_data_mapper['train']['X'],
    lst_y_paths=dct_data_mapper['train']['y'],
    subfolder_path=DATA_SUBFOLDER_PATH,
    batch_size=BATCH_SIZE,
    emb_approximator=emb_approximator,
    fixed_length=FIXED_LENGTH
  )
  dev_dataset, dev_steps_per_epoch = dataset(
    log=log,
    lst_X_paths=dct_data_mapper['dev']['X'],
    lst_y_paths=dct_data_mapper['dev']['y'],
    subfolder_path=DATA_SUBFOLDER_PATH,
    batch_size=BATCH_SIZE,
    emb_approximator=emb_approximator,
    fixed_length=FIXED_LENGTH
  )

  for model_def in model_defs:
    model = get_model(
      log=log,
      input_shape= [FIXED_LENGTH, emb_approximator.emb_size],
      nr_outputs= len(dct_lbls),
      **model_def
    )
    history = train_loop(
      log=log,
      model=model,
      train_dataset=train_dataset,
      steps_per_epoch=train_steps_per_epoch,
      dev_dataset=dev_dataset,
      batch_size=BATCH_SIZE,
      n_epochs=NR_EPOCHS,
      eval_callback=evaluate_callback,
      save_model_callback=save_model_callback
    )
    ### save history



  # train

