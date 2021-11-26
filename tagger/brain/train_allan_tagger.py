from tagger.brain.base_engine import ALLANTaggerEngine 
from tagger.brain.allan_tagger_creator import ALLANTaggerCreator
import tensorflow as tf
import numpy as np
from libraries.lummetry_layers.gated import GatedDense
from collections import OrderedDict
from libraries.logger import Logger
from tagger.brain.data_loader import ALLANDataLoader
import pandas as pd
import os

if __name__ == '__main__':
  root_configs = 'tagger/brain/configs/20190918'
  configs = os.listdir(root_configs)
  
  results = OrderedDict({'MODEL': [], "MAX": [], "EP":[], 'EP_NZ': [] ,'END_SC': [], 'HISTORY': [] })
  
  VALIDATION = True
  
  for i,cfg in enumerate(configs):
    if cfg != 'config_v3_with_v2_emb_noi.txt':
      continue
    
    path_cfg = os.path.join(root_configs, cfg)
    
    l = Logger(lib_name="ALNT",config_file=path_cfg)
    l.SupressTFWarn()
    l.P("*" * 80)
    l.P("")
    l.P("Running iteration {}/{} - '{}'".format(i+1, len(configs), cfg))
    l.P("")
    l.P("*" * 80)

    loader = ALLANDataLoader(log=l, multi_label=True, 
                             normalize_labels=False)
    loader.LoadData()
    
    valid_texts, valid_labels = None, None
    if VALIDATION:
      valid_texts, valid_labels = l.LoadDocuments(folder=l.GetDataSubFolder(l.config_data['TRAINING']['FOLDER']),
                                             doc_ext='.txt',
                                             label_ext='.txt',
                                             doc_folder='Extended_Validation_Texts',
                                             label_folder='Extended_Validation_Labels',
                                             return_labels_list=False)
  
    
    epochs = 150
    
    model_def = l.config_data['MODEL']
    model_name = model_def['NAME']
    eng = ALLANTaggerCreator(log=l, 
                             dict_word2index=loader.dic_word2index,
                             dict_label2index=loader.dic_labels)
    
    if VALIDATION:
      eng.check_labels_set(valid_labels)
    
    eng.setup_model(dict_model_config=model_def, model_name=model_name) # default architecture
    
    hist = eng.train_on_texts(loader.raw_documents,
                              loader.raw_labels,
                              n_epochs=epochs,
                              convert_unknown_words=True,
                              save=True,
                              X_texts_valid=valid_texts,
                              y_labels_valid=valid_labels,
                              skip_if_pretrained=False,
                              DEBUG=False,
                              compute_topic=True)
    
    if VALIDATION:
      score = eng.test_model_on_texts(valid_texts, valid_labels, record_trace=False)
    
      hist, hist_topics = hist
      
      max_idx = np.argmax(hist)
      max_epoch = eng.train_recall_history_epochs[max_idx]
      max_score = hist[max_idx]
      nz_epochs = eng.train_recall_non_zero_epochs
    
      results['MODEL'].append(model_name)
      results['END_SC'].append(score)
      results['HISTORY'].append(hist[-10:])
      results['MAX'].append(max_score)
      results['EP'].append(max_epoch)
      results['EP_NZ'].append(nz_epochs)
      df = pd.DataFrame(results).sort_values('MAX')    
      l.P("")
      l.P("Results so far:\n{}".format(df))
      l.P("")
      l.SaveDataFrame(df, fn='20190919_results1')
  
  