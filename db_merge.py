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
@description: script to save merge corpus
@created on: Fri Nov 26 12:24:06 2021   
@created by: mihai.masala
"""
import numpy as np
import sys
from collections import deque
import time
import csv

from libraries import Logger
from libraries.db_conn.odbc_conn import ODBCConnector

from utils.utils import raw_text_to_words, clean_words_list

qry1= 'select id_document_a, id_document_p, id_paragraf_a, id_paragraf_p, id_paragraf_t, data_start, id_pozitie, id_pozitie_modificatoare from relatie WHERE id_paragraf_p is not NULL and id_paragraf_t is not NULL'
qry2= 'select * from arbori where id_document={} and data_vig=\'{}\' order by idx'
qry3= 'select DISTINCT(data_vig) from arbori where id_document={} order by data_vig DESC'
qry4= 'select id, continut from paragraf where id = {}'

# here only because we might need it later
def old():
    data_start = df_docs.iloc[idx, 5]
      
    doc_a = conn.get_data(sql_query=qry2.format(id_doc_a, data_start))
    doc_p = conn.get_data(sql_query=qry2.format(id_doc_p, data_start))

    text_a, xa = get_text(doc_a, "a", id_paragraf_a)
    text_p, xp = get_text(doc_p, "p", id_paragraf_t)
        
    # get prev data for doc_p
    dates = conn.get_data(sql_query=qry3.format(id_doc_p))
    print(dates)
    prev_text_p, _ = get_text(conn.get_data(sql_query=qry2.format(id_doc_p, dates.iloc[1,0])))
      
    print(len(text_a), text_a)
    print(len(text_p), text_p)
    print(len(prev_text_p), prev_text_p)
    print()
    #print(text_p[-5:])
    print("Modificator:", xa)
    print("Modificat:", xp)


def get_text(query_result, a_or_p=None, id_par=None):
  text = []
  x = None
  for i in range(query_result.shape[0]):
     aux = conn.get_data(sql_query=qry4.format(query_result.iloc[i,3]))
     flags = []
     
     if id_par != None and aux.iloc[0,0] == id_par:
        if a_or_p == "a":
            flags = ["modificator"]
            x = aux.iloc[0,1]
        elif a_or_p == "p":
            flags = ["modificat"]
            x = aux.iloc[0,1]
            
     text.append([aux.iloc[0,0], flags])
     #text.append([aux.iloc[0,1], flags])
  
  return text, x
    

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


  conn = ODBCConnector(log=log, verbose=False, config=config)
  conn.connect(nr_retries=5)
  
  df_docs = conn.get_data(sql_query=qry1)
  print(df_docs.shape[0])
  count = 0
  
  active = set([])
  lens_a = []
  lens_p = []
  lens_t = []

  with open("merge_examples.csv", "w", encoding="utf-8") as outfile:
      writer = csv.writer(outfile)
      writer.writerow(["ID_Pasiv", "Pasiv", "ID_Activ", "Activ", "ID_Transformat", "Transformat"])

      for idx in list(range(df_docs.shape[0])):
      
          if idx % 50000 == 0:
            print(idx)
          #id_doc_a = df_docs.iloc[idx,0]
          #id_doc_p = df_docs.iloc[idx,1]
          
          id_paragraf_a = df_docs.iloc[idx, 2]
          id_paragraf_p = df_docs.iloc[idx, 3]
          id_paragraf_t = df_docs.iloc[idx, 4]
          
          
          a = conn.get_data(sql_query=qry4.format(id_paragraf_a))
          text_paragraf_a = a.iloc[0, 1]
          
          p = conn.get_data(sql_query=qry4.format(id_paragraf_p))
          text_paragraf_p = p.iloc[0, 1]
          
          t = conn.get_data(sql_query=qry4.format(id_paragraf_t))
          text_paragraf_t = t.iloc[0, 1]
          
          #or len(text_paragraf_p) > 600  or len(text_paragraf_a) > 485  or len(text_paragraf_t) > 494:
          
          #if len(text_paragraf_p) < 4 or len(text_paragraf_a) < 4 or len(text_paragraf_t) < 4 or text_paragraf_a[-1] == ":" \
          #         or len(text_paragraf_p) > 340  or len(text_paragraf_a) > 356  or len(text_paragraf_t) > 374:
          #  continue
           
          #if id_paragraf_a in active:
          #  continue
            
          text_paragraf_p = text_paragraf_p.replace("\n", "<NL>")
          text_paragraf_a = text_paragraf_a.replace("\n", "<NL>")
          text_paragraf_t = text_paragraf_t.replace("\n", "<NL>")
          #print()
          #print()
          #print(id_paragraf_p, id_paragraf_a, id_paragraf_t)
          #print("P:", text_paragraf_p)
          #print("A:", text_paragraf_a)
          #print("T:", text_paragraf_t)
          writer.writerow([id_paragraf_p, text_paragraf_p, id_paragraf_a, text_paragraf_a, id_paragraf_t, text_paragraf_t])
        
          lens_a.append(len(text_paragraf_a))
          lens_p.append(len(text_paragraf_p))
          lens_t.append(len(text_paragraf_t))
          
          active.add(id_paragraf_a)
          count += 1
      
  print()
  print("P:", np.min(lens_p), np.mean(lens_p), np.mean(lens_p), np.max(lens_p), np.percentile(lens_p, 90))
  print("A:", np.min(lens_a), np.mean(lens_a), np.mean(lens_a), np.max(lens_a), np.percentile(lens_a, 90))
  print("T:", np.min(lens_t), np.mean(lens_t), np.mean(lens_t), np.max(lens_t), np.percentile(lens_t, 90))
  print("Total:", count)

  