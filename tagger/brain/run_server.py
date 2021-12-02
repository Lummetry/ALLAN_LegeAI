# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 21:14:55 2019

@author: Andrei
"""

from libraries import Logger
from libraries.model_server.simple_model_server import SimpleFlaskModelServer

from tagger.brain.base_engine import ALLANTaggerEngine

import numpy as np

from datetime import datetime


if __name__ == '__main__':
  cfg1 = "tagger/brain/configs/20190918/config_v3_with_v2_emb_noi.txt"  
  l = Logger(lib_name="ALNT",config_file=cfg1, HTML=True)
  l.SupressTFWarn()
  
  eng = ALLANTaggerEngine(log=l,)
  
  eng.initialize()
  
  DEBUG_MODE = 0
  
  TOP = 5
    
  #####      
  # now load FlaskServer tools and .run()
  #####
  def input_callback(data):
    if 'current_query' not in data.keys():
      return None
    s = data['current_query']
    return s
  
  def output_callback(data):
    DEBUG = False
    res1 = data[0]
    topic_document = data[1]
    std_input = data[2]
    enc_input = data[3]
    vals = [x for x in res1.values()]
    keys = [x for x in res1.keys()]    
    top_idxs = np.argsort(vals)[::-1]
    dic_top_best ={keys[x]:float(round(vals[x],3)) for x in top_idxs[:TOP]}
    dic_top_runner ={keys[x]:float(round(vals[x],3)) for x in top_idxs[TOP:]}

    str_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:22]
    
    dct_info = {
        'input_document_init' : std_input,
        'topic_document': topic_document,
        'date_time' : str_now,
        }
    
    if DEBUG:
      dct_info['input_document_post'] = enc_input
      dct_info['runner_tags'] = dic_top_runner
      dct_info['best_tags'] = dic_top_best
      dct_info['comment'] = 'id_topic_document == -1 means ALLANTagger is in DEBUG(0) mode => no topics are available. Switch to DEBUG(1) or NODEBUG.'


    dct_res = {'result' : dct_info}
    return dct_res
  
  
  simple_server = SimpleFlaskModelServer(model=eng,
                                         predict_function='predict_text',
                                         fun_input=input_callback,
                                         fun_output=output_callback,
                                         log=l,
                                         host=l.config_data['HOST'],
                                         port=l.config_data['PORT'])
  simple_server.run()
  
  # now we ca run
  # http://127.0.0.1:5001/analyze?current_query=ala+bala+portocala
    
