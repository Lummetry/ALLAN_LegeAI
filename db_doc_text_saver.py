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
@description: script for saving documents (text) and tags (labels); used for tags corpus
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

def generate_data(debug = False, debug_save_count = 3500, source="from_db"):

  log = Logger(
    lib_name='DBSV', base_folder='.', app_folder='_cache',
    TF_KERAS=False
  )

  config = {
    'CONNECT_PARAMS' : {
      'DRIVER' : '{ODBC Driver 17 for SQL Server}',
      'SERVER' : '195.60.78.50',
      'PORT' : 1433,
      #'DATABASE' : 'LegeV_New',
      'DATABASE' : 'legeV',
      'Uid' : 'damian',
      'Pwd' : '4Esoft1234!@#$2021',
    },

    'QUERY_PARAMS' : None
  }

  qry_docs = 'SELECT distinct vw4.id_document \
              FROM        (SELECT     legeV.dbo.entitate_x_tematica.id_document, legeV.dbo.entitate_x_tematica.id_tip_tematica, legeV.dbo.paragraf.id AS id_paragraf, legeV.dbo.paragraf.continut, DATALENGTH(legeV.dbo.paragraf.continut) AS txt_size \
                   FROM        legeV.dbo.entitate_x_tematica LEFT OUTER JOIN \
                                     legeV.dbo.paragraf ON legeV.dbo.entitate_x_tematica.id_document = legeV.dbo.paragraf.id_document \
                   WHERE     (legeV.dbo.entitate_x_tematica.id_document IN \
                                         (SELECT     id_document \
                                          FROM        (SELECT     id_document, COUNT(id_tip_tematica) AS cnt_tematica \
                                                             FROM        (SELECT     id_document, id_tip_tematica \
                                                                                FROM        legeV.dbo.entitate_x_tematica AS entitate_x_tematica_2 \
                                                                                WHERE     (id_tip_tematica IN \
                                                                                                      (SELECT     id_tip_tematica \
                                                                                                       FROM        (SELECT     id_tip_tematica, COUNT(id_document) AS cnt \
                                                                                                                          FROM        legeV.dbo.entitate_x_tematica AS entitate_x_tematica_1 \
                                                                                                                          GROUP BY id_tip_tematica) AS vw1 \
                                                                                                       WHERE     (cnt > 1000)))) AS vw3 \
                                                             GROUP BY id_document) AS vw4_1 \
                                          WHERE     (cnt_tematica > 1)))) AS vw4 \
                WHERE     vw4.txt_size > 100'

  qry_txt = 'select continut from paragraf where id_document={}'
  qry_lbl = """
  select tip_tematica.nume2 from 
  entitate_x_tematica, tip_tematica 
  where tip_tematica.id=entitate_x_tematica.id_tip_tematica and id_document={}  
  """

  conn = ODBCConnector(log=log, verbose=False, config=config)
  conn.connect(nr_retries=5)

  if source == "from_db":
    df_docs = conn.get_data(sql_query=qry_docs)
  
  elif source.endswith(".csv"):
    df_docs = pd.read_csv(source)

  
  lst_X_docs = []
  lst_y_labels = []
  unique_labels = set()
  log.P("Running params: {}. Debug mode {}".format(sys.argv, "ON" if debug else "OFF"))
  n_iters = df_docs.shape[0]
  timings = deque(maxlen=10)
  print(n_iters)
  total = 0
  
  for idx_doc in range(n_iters):
    
    t0 = time.time()
    id_doc = df_docs.iloc[idx_doc,0]
    
    # process text
    df_text = conn.get_data(sql_query=qry_txt.format(id_doc))
       
    lst_doc_txt = []
    for idx_txt in range(df_text.shape[0]):
      txt = df_text.iloc[idx_txt,0]
      lst_doc_txt.append(txt)

    raw_doc_str = " ".join(lst_doc_txt)
    doc_str = raw_text_to_words(raw_doc_str, max_len=15)
    
    # process labels
    df_labels = conn.get_data(sql_query=qry_lbl.format(id_doc))
    lst_raw_labels = [df_labels.iloc[iii, 0] for iii in range(df_labels.shape[0])]
    lst_labels = clean_words_list(lst_raw_labels)
    for lbl in lst_labels:
      unique_labels.add(lbl)
      
    if len(doc_str) == 0 or len(lst_labels) == 0:
        continue

    
    lst_X_docs.append(doc_str)
    lst_y_labels.append(lst_labels)

    lap_time = time.time() - t0
    timings.append(lap_time)
    mean_time = np.mean(timings)
    remaining_time = (n_iters - (idx_doc + 1)) * mean_time
    if (idx_doc % 100) == 0:
      print("\rProcessed {}/{} documents ({:.1f}%). Remaining time {:.0f}s/{} ({:.1f}s/doc\r".format(
        idx_doc+1, n_iters, 
        (idx_doc+1) / df_docs.shape[0] * 100, 
        remaining_time,
        time.strftime("%H:%M:%S", time.gmtime(remaining_time)),
        mean_time
        ),
        end='', flush=True)    
    
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
       
    if debug and idx_doc > debug_save_count:
        break
      
  print(total)
  print(len(lst_X_docs), len(lst_y_labels))
  lens = [len(x) for x in lst_X_docs]  
  log.P("Obtained {} documents:".format(len(lst_X_docs)))
  log.show_text_histogram(lens, show_both_ends=True, caption='Words per document')
  log.P("Hist:\n{}".format(np.histogram(lens)))
  data = log.save_pickle(
    data=lst_X_docs,
    fn='x_data.pkl',
    folder='data',
    use_prefix=True,
    )

  labels = log.save_pickle(
    data=lst_y_labels,
    fn='y_data.pkl',
    folder='data',
    use_prefix=True,
    )  
  
  n_labels = [len(x) for x in lst_y_labels]
  
  dct_labels = {k:v for v,k in enumerate(unique_labels)}
  log.P("Obtained {} labels:".format(len(dct_labels)))
  log.show_text_histogram(n_labels, show_both_ends=True, caption='Labels per observation')

  dict_label = log.save_pickle(
    data=dct_labels,
    fn='labels_dict.pkl',
    folder='data',
    use_prefix=True,
    )  
  
  return data, labels, dict_label

if __name__ == '__main__':
  
  pass
  #data_file, labels_file, dict_file = generate_data_from_db(debug=True, debug_save_count=1000)
  #print(data_file)
  #print(labels_file)
  #print(dict_file)
  


  