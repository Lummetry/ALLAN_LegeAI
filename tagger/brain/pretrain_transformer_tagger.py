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
from datasets import load_dataset

NUM_PROC = 16
GPU_INDEX_FOR_TRAINING = 3

parser = argparse.ArgumentParser(description='Pre-training for transformer-based document-level multilabel classifier')
parser.add_argument('-data_path', help='path + files desciptor (i.e. _cache/_data/20220209_211238)')
parser.add_argument('-model_path', help='path where to save model (in train mode) or path from where to load the model (in eval mode)')
parser.add_argument('-bert_model', help='path (local) of huggingface name of bert backbone')
parser.add_argument('-bert_max_seq_len', type=int, help='Bert maximum sequence length', default=64)
parser.add_argument('-batch_size', type=int, help='training batch size', default=4)
parser.add_argument('-learning_rate', type=float, help='Learning rate use for training the model', default=1e-5)
parser.add_argument('-epochs', type=int, help='number of training epochs', default=5)
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

    lm_dataset = dataset.map(tokenize_and_chunk, batched=True, num_proc=NUM_PROC, remove_columns=dataset["train"].column_names)

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

    if GPU_INDEX_FOR_TRAINING != -1:
        str_device = '/device:GPU:{}'.format(GPU_INDEX_FOR_TRAINING)
    else:
        str_device = '/device:CPU:0'

    with tf.device(str_device):
        bert_model = TFBertForMaskedLM.from_pretrained(args.bert_model)

        if args.run_type == "train":
            # load and save tokenizer due to bug (loading tokenizer and saving it lead to different indexes)
            tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=False)
            tokenizer.save_pretrained("test_save_model")
            tokenizer = AutoTokenizer.from_pretrained("test_save_model", use_fast=False)

        elif args.run_type == "eval":
            tokenizer = AutoTokenizer.from_pretrained(args.bert_model, use_fast=False)


        train_dataset, test_dataset = load_data(args.data_path, tokenizer)


        optimizer = AdamWeightDecay(learning_rate=args.learning_rate, weight_decay_rate=0.01)
        bert_model.compile(optimizer=optimizer)

        if args.run_type == "train":
            bert_model.fit(x=train_dataset, validation_data=test_dataset, epochs=args.epochs)
            bert_model.evaluate(test_dataset)
            bert_model.save_pretrained("test_save_model")

        elif args.run_type == "eval":
            bert_model.evaluate(test_dataset)
    #endwith
