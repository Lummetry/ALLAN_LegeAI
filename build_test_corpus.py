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
from copyreg import pickle
import numpy as np
import sys
from collections import deque
import time
import pandas as pd
import os
from shutil import copyfile


from libraries import Logger
import pickle as own_pickle

from utils.utils import raw_text_to_words, clean_words_list

def generate_data(csv_file, debug=False):
  log = Logger(
    lib_name='DBSV', base_folder='.', app_folder='_cache',
    TF_KERAS=False
  )

  folder = os.path.join('_cache', "_data")
  files = os.listdir(folder)

  if csv_file == "test_samples_tags.csv":
    tags_files = list(filter(lambda x: x.startswith("tags_v"), files))
    versions = list(map(lambda x: int(x.split("_")[1][1:]), tags_files))
    versions = list(set(versions))
    versions.sort()
    dict_path = "_cache/_data/tags_v{0}_labels_dict.pkl".format(versions[-1])

  
  elif csv_file == "test_samples_qa.csv":
    tags_files = list(filter(lambda x: x.startswith("tags_titles_v"), files))    
    versions = list(map(lambda x: int(x.split("_")[2][1:]), tags_files))
    versions = list(set(versions))
    versions.sort()
    dict_path = "_cache/_data/tags_titles_v{0}_labels_dict.pkl".format(versions[-1])

  with open(dict_path, "rb") as f:
    tags_labels = set(own_pickle.load(f).keys())
  
  print("Number of different labels:", len(tags_labels))

  df_docs = pd.read_csv(csv_file)
  
  lst_X_docs = []
  lst_y_labels = []
  unique_labels = set()
  # log.P("Running params: {}. Debug mode {}".format(sys.argv, "ON" if debug else "OFF"))
  n_iters = df_docs.shape[0]
  timings = deque(maxlen=10)
  rejected_labels = set()

  for idx_doc in range(n_iters):
    t0 = time.time()
    df_text = df_docs.iloc[idx_doc, 0]
    
    # process text
    lst_doc_txt = [df_text]
    raw_doc_str = " ".join(lst_doc_txt)
    doc_str = raw_text_to_words(raw_doc_str, max_len=15)    

    # process labels
    df_labels = df_docs.iloc[idx_doc, 1]
    lst_raw_labels = df_labels.split(",")
    lst_raw_labels = list(map(lambda x: x.strip(), lst_raw_labels))
    lst_labels = clean_words_list(lst_raw_labels)

    if len(doc_str) == 0 or len(lst_labels) == 0:
        continue


    for lbl in lst_labels:
      if lbl not in tags_labels:
        rejected_labels.add(lbl)
        continue
      unique_labels.add(lbl)
    
    new_lst_labels = []
    for lbl in lst_labels:
      if lbl not in rejected_labels:
        new_lst_labels.append(lbl)
    lst_labels = new_lst_labels

    lst_X_docs.append(doc_str)
    lst_y_labels.append(lst_labels)

    lap_time = time.time() - t0
    timings.append(lap_time)
    mean_time = np.mean(timings)
    remaining_time = (n_iters - (idx_doc + 1)) * mean_time
    # if (idx_doc % 10) == 0:
    #   print("\rProcessed {}/{} documents ({:.1f}%). Remaining time {:.0f}s/{} ({:.1f}s/doc\r".format(
    #     idx_doc+1, n_iters, 
    #     (idx_doc+1) / df_docs.shape[0] * 100, 
    #     remaining_time,
    #     time.strftime("%H:%M:%S", time.gmtime(remaining_time)),
    #     mean_time
    #     ),
    #     end='', flush=True)    
    
    if ((idx_doc + 1) % 100000) == 0:
      log.save_pickle(
        data=lst_X_docs,
        fn='x_data_{}.pkl'.format((idx_doc + 1) // 1000000),
        folder='data',
        use_prefix=True,
        )
    
      log.save_pickle(
        data=lst_y_labels,
        fn='y_data_{}.pkl'.format((idx_doc + 1) // 1000000),
        folder='data',
        use_prefix=True,
        )  
     
  lens = [len(x) for x in lst_X_docs]  
  log.P("Obtained {} documents:".format(len(lst_X_docs)))
  # log.show_text_histogram(lens, show_both_ends=True, caption='Words per document')
  # log.P("Hist:\n{}".format(np.histogram(lens)))
  data = log.save_pickle(
    data=lst_X_docs,
    fn='x_data.pkl',
    folder='data',
    use_prefix=True,
    verbose = False
    )

  labels = log.save_pickle(
    data=lst_y_labels,
    fn='y_data.pkl',
    folder='data',
    use_prefix=True,
    verbose = False
    )  
  
  n_labels = [len(x) for x in lst_y_labels]
  
  dct_labels = {k:v for v,k in enumerate(unique_labels)}
  log.P("Obtained {} labels:".format(len(dct_labels)))
  # log.show_text_histogram(n_labels, show_both_ends=True, caption='Labels per observation')

  dict_label = log.save_pickle(
    data=dct_labels,
    fn='labels_dict.pkl',
    folder='data',
    use_prefix=True,
    verbose = False
    )  
  
  copyfile(dict_path, dict_label)

  log.P("Labels not found in dict: {0}".format(rejected_labels))

  return data, labels, dict_label
  
def build_corpus(csv_file):

  data_file, labels_file, dict_file = generate_data(csv_file)


  task = csv_file[len("test_samples_"):][:-len(".csv")]
  if task != "qa" and task != "tags":
    print("Task not recognized {0}".format(task))
    print("Expected task to be <qa> or <tags>. File must be named test_samples_<task>.csv")
    sys.exit()

  files = [data_file, labels_file, dict_file]

  for index, file in enumerate(files):
      file_parts = file.split("\\")
      file_parts[-1] = "test_corpus_{0}".format(task) + "_" + "_".join(file_parts[-1].split("_")[2:])
      new_file = "\\".join(file_parts)

      try:
        os.rename(file, new_file)
      except FileExistsError:
        os.remove(new_file)
        os.rename(file, new_file)

      files[index] = new_file

if __name__ == '__main__':

  pass
  
    