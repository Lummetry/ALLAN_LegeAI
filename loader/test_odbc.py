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
@created on: Thu Dec  2 19:51:43 2021
@created by: damia
"""

from libraries import Logger
from libraries.db_conn.odbc_conn import ODBCConnector 

if __name__ == '__main__':


  log = Logger(
    lib_name='DBTST', base_folder='.', app_folder='_cache',
    TF_KERAS=False
  )

  config = {
    'CONNECT_PARAMS' : {
      'DRIVER' : '{ODBC Driver 17 for SQL Server}',
      'SERVER' : 'cloudifiersql1.database.windows.net',
      'PORT' : 1433,
      'DATABASE' : 'operational',
      'Uid' : 'damian',
      'Pwd' : 'MLteam2021!',
      'Encrypt' : 'yes',
      'TrustServerCertificate' : 'no',
      'Connection Timeout': 30,
    },

    'QUERY_PARAMS' : {
      'default' : {
        'table_data' : 'Invoices',
        'sql_query' : "", ### custom sql query on 'TABLE_DATA' (groupby etc etc); if empty it uses a default sql query
        'chunksize' : 200, ### if removed, then the generator conn.data_chunk_generator() will have only one step
      },

      'default2' : {
        'table_data' : 'Invoices',
        "sql_query" : "",
        'chunksize' : 200,
      }
    }
  }

  conn = ODBCConnector(log=log, config=config)
  conn.connect(nr_retries=5)
  dct_data = conn.get_all_readers_data()  
  df_data = conn.get_data(sql_query='select * from invoices')