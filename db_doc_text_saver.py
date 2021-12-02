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
      'DATABASE' : 'LegeV_New',
      'Uid' : 'damian',
      'Pwd' : '4Esoft1234!@#$2021',
    },

    'QUERY_PARAMS' : None
  }
  
  qry_docs = 'select distinct id_document  from vw_docs'
  qry_txt = 'select continut from paragraf where id_document={}'
  qry_lbl = """
  select tip_tematica.nume2 from 
  entitate_x_tematica, tip_tematica 
  where tip_tematica.id=entitate_x_tematica.id_tip_tematica and id_document={}  
  """

  conn = ODBCConnector(log=log, verbose=False, config=config)
  conn.connect(nr_retries=5)
  
  df_docs = conn.get_data(sql_query=qry_docs)

  lst_X_docs = []
  lst_y_labels = []
  for idx_doc in range(df_docs.shape[0]):
    id_doc = df_docs.iloc[idx_doc,0]
    
    # process text
    df_text = conn.get_data(sql_query=qry_txt.format(id_doc))
    lst_doc_txt = []
    for idx_txt in range(df_text.shape[0]):
      txt = df_text.iloc[idx_txt,0]
      lst_doc_txt.append(txt)
    raw_doc_str = " ".join(lst_doc_txt)
    doc_str = raw_text_to_words(raw_doc_str)
    
    
    # process labels
    df_labels = conn.get_data(sql_query=qry_lbl.format(id_doc))
    lst_raw_labels = [df_labels.iloc[iii, 0] for iii in range(df_labels.shape[0])]
    lst_labels = clean_words_list(lst_raw_labels)
    
    lst_X_docs.append(doc_str)
    lst_y_labels.append(lst_labels)

    if (idx_doc % 100) == 0:
      print("\rProcessing document {}/{} ({:1f}%): \r".format(
        idx_doc+1, df_docs.shape[0], 
        (idx_doc+1) / df_docs.shape[0] * 100, ),
        end='', flush=True)    
    break
  
  
  print(raw_doc_str)
  print('*'*80)
  print(lst_raw_labels)
  print('*'*80)
  print('*'*80)
  print(doc_str)
  print('*'*80)
  print(lst_labels)
  print(lst_raw_labels)
    
    
    
    