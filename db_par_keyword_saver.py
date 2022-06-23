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
@created on: Fri Nov 26 12:24:06 2021
@created by: damia
"""
import numpy as np
import sys
from collections import deque
import time

from libraries import Logger
from libraries.db_conn.odbc_conn import ODBCConnector

from utils.utils import raw_text_to_words, clean_words_list

if __name__ == '__main__':
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
  
  config2 = {
    'CONNECT_PARAMS' : {
      'DRIVER' : '{ODBC Driver 17 for SQL Server}',
      'SERVER' : '195.60.78.50',
      'PORT' : 1433,
      'DATABASE' : 'LegeV_New',
      'Uid' : 'damian',
      'Pwd' : '4Esoft1234!@#$2021',
    },

    'QUERY_PARAMS' : None
  }

  
  qry_docs = 'select distinct id_document  from vw_docs'
  #qry_pars = 'select distinct id_paragraf from paragraf_x_cuvant_cheie'
  qry_pars = 'SELECT DISTINCT [ID_PARAGRAF]  FROM [legeV].[dbo].[paragraf_x_cuvant_cheie] where ID_PARAGRAF in (select id from paragraf where id_document in (10941734, 11002879))'
  qry_txt = 'select continut from paragraf where id={}'
  qry_lbl = 'select id_cuvant_cheie from paragraf_x_cuvant_cheie where ID_PARAGRAF = {}'
  qry_l   = 'select nume from tip_cuvantcheie where uid = {}'

  conn = ODBCConnector(log=log, verbose=False, config=config)
  conn.connect(nr_retries=5)
  
  conn2 = ODBCConnector(log=log, verbose=False, config=config2)
  conn2.connect(nr_retries=5)
  
  df_pars = conn.get_data(sql_query=qry_pars)
  #print(df_pars)
  
  lst_X_pars = []
  lst_y_labels = []
  unique_labels = set()
  DEBUG = len(sys.argv) > 1 and sys.argv[1].upper() == 'DEBUG'
  log.P("Running params: {}. Debug mode {}".format(sys.argv, "ON" if DEBUG else "OFF"))
  n_iters = df_pars.shape[0]
  print(n_iters)
  timings = deque(maxlen=10)
  for idx_par in range(n_iters):
    t0 = time.time()
    id_par = df_pars.iloc[idx_par,0]
    #print(id_par)
    #if id_par in [457353995, 457354006, 457354017, 457354028, 457354039, 457354050, 457354061, 457354072, 457354083, 457354094]:
    #    continue
    
    # process text
    df_text = conn.get_data(sql_query=qry_txt.format(id_par))
    if df_text.shape[0] > 1:
        print("More than one entry for par", id_par)
        sys.exit()
    elif df_text.shape[0] == 0:
        continue
    txt = df_text.iloc[0, 0]
    #print(txt)
    par_str = raw_text_to_words(txt, max_len=15)    
    if len(par_str) < 1:
        continue
    #print(par_str)
    # process labels
   
    lst_raw_labels = []
    df_keyword_ids = conn.get_data(sql_query=qry_lbl.format(id_par))
    
    for idx_label in range(df_keyword_ids.shape[0]):
        keyword_id = df_keyword_ids.iloc[idx_label,0]
        #print(id_par, keyword_id)
        keyword_label = conn2.get_data(sql_query=qry_l.format(keyword_id)).iloc[0,0]
        lst_raw_labels.append(keyword_label)
    #print(lst_raw_labels)
   
    if df_keyword_ids.shape[0] > 1:
        print(id_par)
        print(par_str)
        print(lst_raw_labels)
        sys.exit()

    #lst_raw_labels = [df_labels.iloc[iii, 0] for iii in range(df_labels.shape[0])]
    lst_labels = clean_words_list(lst_raw_labels)

    for lbl in lst_labels:
      unique_labels.add(lbl)

    lst_X_pars.append(par_str)
    lst_y_labels.append(lst_labels)

    lap_time = time.time() - t0
    timings.append(lap_time)
    mean_time = np.mean(timings)
    remaining_time = (n_iters - (idx_par + 1)) * mean_time
    if (idx_par % 10) == 0:
      print("\rProcessed {}/{} documents ({:.1f}%). Remaining time {:.0f}s/{} ({:.1f}s/doc\r".format(
        idx_par+1, n_iters, 
        (idx_par+1) / df_pars.shape[0] * 100, 
        remaining_time,
        time.strftime("%H:%M:%S", time.gmtime(remaining_time)),
        mean_time
        ),
        end='', flush=True)    
  
  lens = [len(x) for x in lst_X_pars]  
  log.P("Obtained {} documents:".format(len(lst_X_pars)))
  log.show_text_histogram(lens, show_both_ends=True, caption='Words per paragraph')
  log.P("Hist:\n{}".format(np.histogram(lens)))
  log.save_pickle(
    data=lst_X_pars,
    fn='x_data.pkl',
    folder='data',
    use_prefix=True,
    )

  log.save_pickle(
    data=lst_y_labels,
    fn='y_data.pkl',
    folder='data',
    use_prefix=True,
    )  
  
  n_labels = [len(x) for x in lst_y_labels]
  
  dct_labels = {k:v for v,k in enumerate(unique_labels)}
  log.P("Obtained {} labels:".format(len(dct_labels)))
  log.show_text_histogram(n_labels, show_both_ends=True, caption='Labels per observation')

  log.save_pickle(
    data=dct_labels,
    fn='labels_dict.pkl',
    folder='data',
    use_prefix=True,
    )  

  