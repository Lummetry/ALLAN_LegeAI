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
*
*
*  RO:
*    Modul software TempRent, proiect finanțat în cadrul POC, Axa prioritara 2 - Tehnologia Informației și Comunicațiilor (TIC) 
*    pentru o economie digitală competitivă, Prioritatea de investiții 2b - Dezvoltarea produselor și s
*    erviciilor TIC, a comerțului electronic și a cererii de TIC, cod SMIS 142474, 
*    Contractul de finanțare nr. 2/221_ap3/24.06.2021.
*


@copyright: Lummetry.AI
@author: Lummetry.AI - Laurentiu
@project: 
@description:
  
  
  RO:
    Acest produs a fost livrat si realizat in baza serviciilor de cercetare-inovare industrială 
    conform contract de servicii nr. 9 din 01.11.2021 folosind modulele AI "ALLAN" aferente "TempRent" -  
    Proiect finanțat în cadrul POC, Axa prioritara 2 - Tehnologia Informației și Comunicațiilor (TIC) 
    pentru o economie digitală competitivă, Prioritatea de investiții 2b - Dezvoltarea produselor și s
    erviciilor TIC, a comerțului electronic și a cererii de TIC, cod SMIS 142474, 
    Contractul de finanțare nr. 2/221_ap3/24.06.2021.
    
"""

import argparse

from libraries import Logger
from libraries.model_server_v2.gateway import FlaskGateway

### Example for running a gateway
if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  parser.add_argument(
    '-b', '--base_folder',
    type=str, default='.',
    help='Logger base folder'
  )

  parser.add_argument(
    '-a', '--app_folder',
    type=str, default='_cache',
    help='Logger app folder'
  )

  parser.add_argument(
    '--host', type=str, default='0.0.0.0'
  )

  parser.add_argument(
    '--port', type=int, default=5002
  )

  args = parser.parse_args()
  base_folder = args.base_folder
  app_folder = args.app_folder
  host = args.host
  port = args.port

  ### Attention! config_file should contain the configuration for each endpoint; 'NR_WORKERS' and upstream configuration
  log = Logger(
    lib_name='GTW',
    config_file='config_gateway.txt',
    base_folder=base_folder, app_folder=app_folder,
    TF_KERAS=False
  )

  gtw = FlaskGateway(
    log=log,
    # server_names=['get_tags', 'get_qa', 'get_sim', 'get_aprox', 'get_conf'],
    workers_location='endpoints',
    workers_suffix='Worker',
    host=host,
    port=port,
    #first_server_port=5020,
    server_execution_path='/run'
  )
