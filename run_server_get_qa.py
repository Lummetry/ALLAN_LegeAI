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
@author: Lummetry.AI - Laurentiu
@project: 
@description:
"""

import argparse

from libraries import Logger
from libraries.model_server_v2 import FlaskModelServer

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
    '--port', type=int, default=5004
  )


  args = parser.parse_args()
  base_folder = args.base_folder
  app_folder = args.app_folder
  port = args.port

  log = Logger(lib_name='SVR', base_folder=base_folder, app_folder=app_folder, TF_KERAS=False)

  svr = FlaskModelServer(
    log=log,
    plugins_location='endpoints',
    plugin_name='get_qa',
    plugin_suffix='Worker',
    port=port,
    nr_workers=5
  )

