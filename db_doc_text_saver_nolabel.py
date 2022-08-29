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
@description: script for saving documents (text); used for saving pretraining texts
@created on: Fri Nov 26 12:24:06 2021
@created by: mihai.masala
"""
import numpy as np
import sys
from collections import deque
import time

from libraries import Logger
from libraries.db_conn.odbc_conn import ODBCConnector

from utils.utils import raw_text_to_words, clean_words_list

if __name__ == '__main__':

  how_often_to_save = 10000
  how_often_to_report = 10
  log = Logger(
    lib_name='DBSV', base_folder='.', app_folder='_cache',
    TF_KERAS=False
  )

  config = {
    'CONNECT_PARAMS' : {
      'DRIVER' : '{ODBC Driver 17 for SQL Server}',
      'SERVER' : '195.60.78.50',
      'PORT' : 1433,
      'DATABASE' : 'LegeV',
      'Uid' : 'damian',
      'Pwd' : '4Esoft1234!@#$2021',
    },

    'QUERY_PARAMS' : None
  }
  
  #qry_docs = 'select TOP (10000) id  from document'
  qry_docs = 'select distinct id  from document'
  qry_txt = 'select continut from paragraf where id_document={}'
  
  conn = ODBCConnector(log=log, verbose=False, config=config)
  conn.connect(nr_retries=5)
  
  df_docs = conn.get_data(sql_query=qry_docs)

  lst_X_docs = []
  lst_y_labels = []
  unique_labels = set()
  DEBUG = len(sys.argv) > 1 and sys.argv[1].upper() == 'DEBUG'
  log.P("Running params: {}. Debug mode {}".format(sys.argv, "ON" if DEBUG else "OFF"))
  n_iters = df_docs.shape[0]
  timings = deque(maxlen=10)
  global_lens = []
  skipped = []
  for idx_doc in range(n_iters):
  
    if idx_doc < 7 * how_often_to_save:
        continue
    t0 = time.time()
    id_doc = df_docs.iloc[idx_doc,0]
    
    # process text
    df_text = conn.get_data(sql_query=qry_txt.format(id_doc))
    if df_text.shape[0] > 1000 or df_text.shape[0] < 4:
        skipped.append([id_doc, df_text.shape[0]])
        continue
    lst_doc_txt = []
    for idx_txt in range(df_text.shape[0]):
      txt = df_text.iloc[idx_txt,0]
      if txt != None:
        lst_doc_txt.append(txt + "pastreaza_cuvant")
    
    raw_doc_str = " ".join(lst_doc_txt)
    doc_str = raw_text_to_words(raw_doc_str, max_len=15)    
    doc_str = list(map(lambda x: x.strip(), doc_str))
    doc_str = " ".join(doc_str).replace("pastreaza_cuvant pastreaza_cuvant ", "\n").replace("pastreaza_cuvant ", "\n").replace(" pastreaza_cuvant", "\n").replace("pastreaza_cuvant", "\n")
    if "pastreaza_cuvant" in doc_str:
        print(doc_str)
        print("pastreaza_cuvant still in text")
        sys.exit()
    
    
    lst_X_docs.append(doc_str)

    lap_time = time.time() - t0
    timings.append(lap_time)
    mean_time = np.mean(timings)
    remaining_time = (n_iters - (idx_doc + 1)) * mean_time
    if (idx_doc % how_often_to_report) == 0:
      print("\rProcessed {}/{} documents ({:.1f}%). Remaining time {:.0f}s/{} ({:.1f}s/doc\r".format(
        idx_doc+1, n_iters, 
        (idx_doc+1) / df_docs.shape[0] * 100, 
        remaining_time,
        time.strftime("%H:%M:%S", time.gmtime(remaining_time)),
        mean_time
        ),
        end='', flush=True)    
    
    if ((idx_doc + 1) % how_often_to_save) == 0:
        with open("_cache/_data/pretraining_texts/texts{0}.txt".format((idx_doc + 1) // how_often_to_save), "w") as save_file:
            for doc in lst_X_docs:
                save_file.writelines(doc)
                save_file.write("\n")
        
        
        lens = [len(x) for x in lst_X_docs]  
        global_lens.extend(lens)    
        lst_X_docs = []
   
  if len(lst_X_docs) > 0:   
    lens = [len(x) for x in lst_X_docs]  
    global_lens.extend(lens)
  log.P("Obtained {} documents:".format(len(global_lens)))
  log.show_text_histogram(lens, show_both_ends=True, caption='Words per document')
  log.P("Hist:\n{}".format(np.histogram(global_lens)))
  print(np.min(global_lens), np.mean(global_lens), np.mean(global_lens), np.max(global_lens), np.percentile(global_lens, 90))


  s = list(map(lambda x: x[1], skipped))
  print("len", len(s), np.min(s), np.mean(s), np.median(s), np.max(s))
  

  
  if len(lst_X_docs) > 0:   
    with open("_cache/_data/pretraining_texts/texts{0}.txt".format((idx_doc + 1) // how_often_to_save), "w") as save_file:
        for doc in lst_X_docs:
            save_file.writelines(doc)
            save_file.write("\n")
  

  