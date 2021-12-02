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

from libraries import Logger
from tagger.brain.emb_aproximator import EmbeddingApproximator
from tagger.brain.paragraph_tagger.train_paragraph_tagger_data import dataset
from tagger.brain.paragraph_tagger.train_paragraph_tagger_models import get_tagger_model


def train_loop(log, train_dataset, test_dataset, batch_size, n_epochs, model,
               test_callback,
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
  n_obs = len(train_dataset)
  log.P("Training on {} obs, {} epochs, batch {}".format(
    n_obs, n_epochs, batch_size))
  n_batches = n_obs // batch_size + 1
  train_losses = []
  log.SupressTFWarn()
  best_recall = 0
  train_recall_history = []
  train_recall_history_epochs = []
  train_recall_non_zero_epochs = []
  train_epoch = 0

  # TODO: eval
  #fct_test = self.test_model_on_texts_with_topic if compute_topic else self.test_model_on_texts

  for epoch in range(n_epochs):
    train_epoch = epoch + 1
    epoch_losses = []
    for i_batch in range(n_batches):
      ### TODO: refactor ###
      batch_start = (i_batch * batch_size) % n_obs
      batch_end = min(batch_start + batch_size, n_obs)
      X_batch = np.array(X_data[batch_start:batch_end].tolist())
      y_batch = np.array(y_data[batch_start:batch_end])
      batch_output = model.train_on_batch(X_batch, y_batch)
      s_bout = log.EvaluateSummary(model, batch_output)
      loss = batch_output[0] if type(batch_output) in [list, np.ndarray, tuple] else batch_output
      batch_info = "Epoch {:>3}: {:>5.1f}% completed [{}]".format(
        epoch + 1, i_batch / n_batches * 100, s_bout)
      print("\r {}".format(batch_info), end='', flush=True)
      train_losses.append(loss)
      epoch_losses.append(loss)
      trained = True
      #########################
    print("\r", end="")
    epoch_loss = np.mean(epoch_losses)
    log.P("Epoch {} done. loss:{:>7.4f}, all avg :{:>7.4f}, last batch: [{}]".format(
      epoch + 1, epoch_loss, np.mean(train_losses), s_bout))
    if (epoch > 0) and (test_every_epochs > 0) and (test_dataset is not None) and (
            (epoch + 1) % test_every_epochs == 0):
      log.P("Testing on epoch {}".format(epoch + 1))
      rec = test_callback(
        test_dataset=test_dataset,
        DEBUG=True,
        top=10
      )

      if compute_topic:
        rec, topic_rec = rec
      if last_test_non_zero and (best_recall < rec):
        train_recall_non_zero_epochs.append(epoch + 1)
        s_name = 'ep{}_R{:.0f}_ANZ'.format(epoch + 1, rec)
        save_model_callback(model, s_name, delete_prev_named=True) ### TODO: CHECK ORIGINAL `save_model`
        best_recall = rec
      elif best_recall < rec:
        s_name = 'ep{}_R{:.0f}'.format(epoch + 1, rec)
        save_model_callback(model, s_name, delete_prev_named=True) ### TODO: CHECK ORIGINAL `save_model`
        best_recall = rec

  log.P("Model training done.")
  log.P("Train recall history: {}".format(train_recall_history))
  if compute_topic:
    log.P("Train topic recall history: {}".format(train_topic_recall_history))

  self._reload_embeds_from_model() ### TODO: Maybe remove

  return train_recall_history

if __name__ == '__main__':
  log = Logger(...)

  EMB_MODEL_NAME = 'ro_embgen_model.h5'

  TRAIN_DATA_FILE = ...
  TEST_DATA_FILE = ...
  DCT_LBL_FN = 'dict_lbl.pk'

  BATCH_SIZE = ...
  NR_EPOCHS = ...

  model_defs = ... # load from config

  dct_lbls = log.load_pickle_from_data(DCT_LBL_FN)

  emb_approximator = EmbeddingApproximator(log=log)
  emb_approximator.dic_labels = dct_lbls
  emb_approximator.maybe_load_pretrained_embgen(embgen_model_file=EMB_MODEL_NAME)
  # emb_approximator._get_generated_embeddings()


  train_dataset = dataset(
    log=log,
    data_file=TRAIN_DATA_FILE,
    batch_size=2,
    emb_approximator=emb_approximator,
    fixed_length=50
  )
  test_dataset = dataset(
    log=log,
    data_file=TEST_DATA_FILE,
    batch_size=2,
    emb_approximator=emb_approximator,
    fixed_length=50
  )

  for model_def in model_defs:
    model = get_tagger_model(**model_def)
    history = train_loop(
      log=log,
      model=model,
      train_dataset=train_dataset,
      test_dataset=test_dataset,
      batch_size=BATCH_SIZE,
      n_epochs=NR_EPOCHS,
      test_callback=lambda x:x,
      save_model_callback=lambda x:x
    )
    ### save history



  # train

