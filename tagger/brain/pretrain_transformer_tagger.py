import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
from transformers import AutoTokenizer, TFBertForMaskedLM
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import sys
import os
import argparse
import pandas as pd
from transformers import DataCollatorForLanguageModeling
from transformers import AdamWeightDecay




parser = argparse.ArgumentParser(description='Training and evaluation file for transformer-based document-level multilabel classifier')
parser.add_argument('-args_path', help='path to config file')
parser.add_argument('-data_path', help='path + files desciptor (i.e. _cache/_data/20220209_211238)')
parser.add_argument('-model_path', help='path where to save model (in train mode) or path from where to load the model (in eval mode)')
parser.add_argument('-bert_model', help='path (local) of huggingface name of bert backbone')
parser.add_argument('-bert_max_seq_len', type=int, help='Bert maximum sequence length', default=64)
parser.add_argument('-batch_size', type=int, help='training batch size', default=4)
parser.add_argument('-learning_rate', type=float, help='Learning rate use for training the model', default=1e-5)
parser.add_argument('-epochs', type=int, help='number of training epochs', default=5)
parser.add_argument('-bert_frozen', default=False, action='store_true', help='flag to mark if the bert backbone if trainable or not')
parser.add_argument('-use_generator', default=False, action='store_true', help='flag to mark the usage of generators rather than loading all data in tf.data.Dataset')
parser.add_argument('-k', nargs='+', help='List of k\'s to use for evaluating metrics @k', default=[1, 3, 5])
parser.add_argument('-run_type', default="train", help='train or eval run')
args = parser.parse_args()

def load_data(data_path, tokenizer):

    train_files, test_files = [], []
    for x in os.listdir(os.path.join(data_path, "train")):
        train_files.append(os.path.join(os.path.join(data_path, "train"), x))

    for x in os.listdir(os.path.join(data_path, "test")):
        test_files.append(os.path.join(os.path.join(data_path, "test"), x))
        
    print(train_files)
    print(test_files)

    dataset = {}
    train_dataset = []
    test_dataset = []

    for train_file in train_files:
        with open(train_file, "r") as f:
            inner_entry = {}
            inner_entry["text"] = []
            text = f.readlines()
            for line in text:
                if line == "\n":
                    inner_entry["text"] = " ".join(inner_entry["text"])
                    train_dataset.append(inner_entry)
                    inner_entry = {}
                    inner_entry["text"] = []                   
                else:
                    inner_entry["text"].append(line)

    for test_file in test_files:
        with open(test_file, "r") as f:
            inner_entry = {}
            inner_entry["text"] = []
            text = f.readlines()
            for line in text:
                if line == "\n":
                    inner_entry["text"] = " ".join(inner_entry["text"])
                    test_dataset.append(inner_entry)
                    inner_entry = {}
                    inner_entry["text"] = []                   
                else:
                    inner_entry["text"].append(line)


    from datasets import load_dataset
    dataset = load_dataset('text', data_files={'train': train_files, 'test': test_files}, encoding="unicode_escape")


    def tokenize_and_chunk(examples):
        all_input_ids = []

        for x in examples["text"]:
            if len(x) < 2:
                all_input_ids.extend(tokenizer("\n")["input_ids"])
                continue

            input_ids = tokenizer(x)["input_ids"]
            all_input_ids.extend(input_ids)
            all_input_ids.append(tokenizer.sep_token_id)
        
        chunks = []
        for idx in range(0, len(all_input_ids), args.bert_max_seq_len):
            if len(all_input_ids[idx: idx + args.bert_max_seq_len]) != args.bert_max_seq_len:
                continue
            chunks.append(all_input_ids[idx: idx + args.bert_max_seq_len])
        
        result = {}
        result["input_ids"] = chunks
        result["labels"] = result["input_ids"].copy()


        return result

    lm_dataset = dataset.map(tokenize_and_chunk, batched=True, num_proc=1, remove_columns=dataset["train"].column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf")

    tf_train_set = lm_dataset["train"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    shuffle=True,
    batch_size=args.batch_size,
    collate_fn=data_collator)

    tf_test_set = lm_dataset["test"].to_tf_dataset(
    columns=["attention_mask", "input_ids", "labels"],
    shuffle=False,
    batch_size=args.batch_size,
    collate_fn=data_collator)

    return tf_train_set, tf_test_set

if __name__ == "__main__":

    bert_model = TFBertForMaskedLM.from_pretrained(args.bert_model)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=False)

    train_dataset, test_dataset = load_data(args.data_path, tokenizer)

    
    optimizer = AdamWeightDecay(learning_rate=5e-6, weight_decay_rate=0.01)
    bert_model.compile(optimizer=optimizer)
    bert_model.fit(x=train_dataset, validation_data=test_dataset, epochs=3)




    sys.exit()