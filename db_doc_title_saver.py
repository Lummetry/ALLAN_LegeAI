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
import numpy as np
import sys
from collections import deque
import time
import pandas as pd

from libraries import Logger
from libraries.db_conn.odbc_conn import ODBCConnector

from utils.utils import raw_text_to_words, clean_words_list, preprocess_title
import spacy


REMOVE_PARAN = 0
REMOVE_PREFIX = 1
REMOVE_POS = 2
REMOVE_STOPWORDS = 3
REMOVE_DEP = 4
REMOVE_NONALPHA = 5
REMOVE_ENTITIES = 6


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
  
  qry_docs = 'select * from \
( \
	select id_document, count(id_tip_tematica) cnt_tematica from \
		( \
			select id_document, id_tip_tematica from  LegeV.[dbo].[entitate_x_tematica] \
			where id_tip_tematica in \
				(select id_tip_tematica from \
					(SELECT     id_tip_tematica, COUNT(id_document) AS cnt \
					FROM        LegeV.dbo.entitate_x_tematica \
					GROUP BY id_tip_tematica \
					) vw1 \
				where vw1.cnt > 1000 \
				) \
		) as vw3 \
	group by vw3.id_document \
) vw4 \
where vw4.cnt_tematica > 1'


  qry_txt = 'select titlu from document where id={}'
  qry_lbl = """
  select tip_tematica.nume2 from 
  entitate_x_tematica, tip_tematica 
  where tip_tematica.id=entitate_x_tematica.id_tip_tematica and id_document={}  
  """

  conn = ODBCConnector(log=log, verbose=False, config=config)
  conn.connect(nr_retries=5)


  df_docs = conn.get_data(sql_query=qry_docs)
  
  nlp = spacy.load('ro_core_news_lg')

  
  if source.endswith(".csv"):
    df_docs_csv = pd.read_csv(source)
    df_docs = pd.concat([df_docs, df_docs_csv], axis=0, ignore_index=True)

  lst_X_docs = []
  lst_y_labels = []
  unique_labels = set()
  log.P("Running params: {}. Debug mode {}".format(sys.argv, "ON" if debug else "OFF"))
  n_iters = df_docs.shape[0]
  timings = deque(maxlen=10)

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

    if len(doc_str) == 0 or len(lst_labels) == 0:
        continue

    title = " ".join(doc_str)
    res = preprocess_title(title, nlp=nlp, proc=[REMOVE_PARAN, REMOVE_PREFIX, REMOVE_POS, REMOVE_DEP, REMOVE_NONALPHA, REMOVE_ENTITIES])
    doc_str = res.split(" ")
        
    if len(doc_str) <= 2 or len(doc_str) > 20:
        continue



    for lbl in lst_labels:
      unique_labels.add(lbl)
    
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