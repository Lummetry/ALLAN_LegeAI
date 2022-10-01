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
@created on: Mon Sep 19 16:56:23 2022
@created by: mihai.masala
"""
import db_doc_text_saver
import db_doc_title_saver
import os
import sys
import subprocess
import argparse
from build_test_corpus import build_corpus

parser = argparse.ArgumentParser()
parser.add_argument('--base_folder', type=str, default="_cache")
parser.add_argument('--task', type=str, required=True)
parser.add_argument('--data_source', type=str, default='from_db')


args = parser.parse_args()


def get_last_tags_dataset_version():
    folder = os.path.join(args.base_folder, "_data")
    files = os.listdir(folder)
    
    tags_files = list(filter(lambda x: x.startswith("tags_v"), files))
    
    versions = list(map(lambda x: int(x.split("_")[1][1:]), tags_files))
    versions = list(set(versions))
    versions.sort()
    
    return versions[-1]

def get_last_qa_dataset_version():
    folder = os.path.join(args.base_folder, "_data")
    files = os.listdir(folder)
    
    tags_files = list(filter(lambda x: x.startswith("tags_titles_v"), files))
    
    versions = list(map(lambda x: int(x.split("_")[2][1:]), tags_files))
    versions = list(set(versions))
    versions.sort()
    
    return versions[-1]

def get_last_tags_model_version():
    folder = os.path.join(args.base_folder, "_models")
    files = os.listdir(folder)
    
    tags_files = list(filter(lambda x: x.startswith("tags_") and "titles" not in x, files))
    
    versions = list(map(lambda x: int(x.split("_")[1]), tags_files))
    versions = list(set(versions))
    versions.sort()
    return versions[-1]

def get_last_qa_model_version():
    folder = os.path.join(args.base_folder, "_models")
    files = os.listdir(folder)
    
    tags_files = list(filter(lambda x: x.startswith("tags_titles"), files))
    
    versions = list(map(lambda x: int(x.split("_")[2]), tags_files))
    versions = list(set(versions))
    versions.sort()
    return versions[-1]

def rename_files_tags(files):

    last_tags_version = get_last_tags_dataset_version()
    
    for index, file in enumerate(files):
        file_parts = file.split("\\")
        file_parts[-1] = "tags_v{0}".format(last_tags_version + 1) + "_" + "_".join(file_parts[-1].split("_")[2:])
        new_file = "\\".join(file_parts)
    
        os.rename(file, new_file)
        
        files[index] = new_file
    
    return files

def rename_files_qa(files):

    last_tags_version = get_last_qa_dataset_version()
    
    for index, file in enumerate(files):
        file_parts = file.split("\\")
        file_parts[-1] = "tags_titles_v{0}".format(last_tags_version + 1) + "_" + "_".join(file_parts[-1].split("_")[2:])
        new_file = "\\".join(file_parts)
    
        os.rename(file, new_file)
        
        files[index] = new_file
    
    return files

def get_data_path_tags(data_file):
    
    return data_file.split("tags_v")[0]+"tags_v"+str(int(data_file.split("tags_v")[1].split("_x")[0]))

def get_data_path_qa(data_file):
    
    return data_file.split("tags_titles_v")[0]+"tags_titles_v"+str(int(data_file.split("tags_titles_v")[1].split("_x")[0]))

def get_user_input():
  
  data_source = input('Enter the source of the data: from the live database (from_db), from a .csv file (path to the file) or from a data folder (like tags_vX for get_tags or tags_tiltes_vX for get qa) [default: from_db]: ')
  if data_source == '':
    data_source = "from_db"
  if data_source != "from_db" and not(data_source.endswith(".csv")) and not os.path.isfile(os.path.join(args.base_folder, "_data/"+data_source+"_labels_dict.pkl")):
    print("Invalid option for data_source")
    input("Press ENTER to exit.")
    sys.exit()
  
  if args.task == "get_tags":
    epochs = 5
    batch_size = 4
    bert_max_seq_len = 512
    
  elif args.task == "get_qa":
    epochs = 17
    batch_size = 32
    bert_max_seq_len = 64
  
  epochs_i = input('Enter the number of training epochs [{0}]: '.format(epochs))
  if epochs_i != '':
    epochs = int(epochs_i)
    
  batch_size_i = input('Enter the batch size [{0}]: '.format(batch_size))
  if batch_size_i != '':
    batch_size = int(batch_size_i)
  
  bert_max_seq_len_i = input('Enter the size of BERT sequnce length (max 512) [{0}]: '.format(bert_max_seq_len))
  if bert_max_seq_len_i != '':
    bert_max_seq_len = int(bert_max_seq_len_i)
  
  return data_source, epochs, batch_size, bert_max_seq_len
  
  
    

if __name__ == "__main__":

    data_source, epochs, batch_size, bert_max_seq_len = get_user_input()
    #region config
    if args.task == "get_tags":
        function_get_data = db_doc_text_saver.generate_data
        rename_files = rename_files_tags
        get_last_model_version = get_last_tags_model_version
        get_data_path = get_data_path_tags
        model_prefix = "tags"

    elif args.task == "get_qa":
        function_get_data = db_doc_title_saver.generate_data
        rename_files = rename_files_qa
        get_last_model_version = get_last_qa_model_version
        get_data_path = get_data_path_qa
        model_prefix = "tags_titles"
    #endregion config
    
    
    #region data
    if data_source == "from_db" or data_source.endswith(".csv"):
        # get data from db
        data_file, labels_file, dict_file = function_get_data(debug=False, debug_save_count = 5000, source=data_source)
        [data_file, labels_file, dict_file] = rename_files([data_file, labels_file, dict_file])
    else:
        data_file = os.path.join(os.path.join(args.base_folder, "_data"), data_source)
    #endregion data

    #region paths
    data_path = get_data_path(data_file)
    model_version = get_last_model_version()
    model_path = args.base_folder + "/_models/{0}_{1}_full".format(model_prefix, model_version + 1)
    #endregion paths
    
    #region test corpus
    build_corpus("test_samples_{0}.csv".format(args.task[len("get_"):]))
    #endregion test corpus
    
    
    #region training
    command = "python tagger\\brain\\train_transformer_tagger.py -use_generator -data_path={0} -k 3 -epochs={1} -learning_rate=1e-5 -bert_max_seq_len={2} -batch_size={3} -bert_backbone=readerbench/jurBERT-base -run_type=train_full -model_path={4} -dev_run=False".format(data_path, epochs, bert_max_seq_len, batch_size, model_path)
    subprocess.run(command)
    #endregion training
    
    #region 
    print("Used corpus:", data_path)
    print("Trained model:", os.path.join(model_path, "weights/epoch{:02d}".format(epochs)))
    #endregion
    
    input("Press ENTER to exit.")
