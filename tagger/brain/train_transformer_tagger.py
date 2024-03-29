import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
import pickle
from transformers import AutoTokenizer, TFBertModel
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import sys
import random
import os
import argparse
import json
import pandas as pd
import copy

GPU_INDEX_FOR_TRAINING = 0


parser = argparse.ArgumentParser(description='Training and evaluation file for transformer-based document-level multilabel classifier')
parser.add_argument('-args_path', help='path to config file')
parser.add_argument('-data_path', help='path + files desciptor (i.e. _cache/_data/20220209_211238)')
parser.add_argument('-model_path', help='path where to save model (in train mode) or path from where to load the model (in eval mode)')
parser.add_argument('-bert_backbone', help='path (local) of huggingface name of bert backbone')
parser.add_argument('-bert_max_seq_len', type=int, help='Bert maximum sequence length', default=64)
parser.add_argument('-batch_size', type=int, help='training batch size', default=4)
parser.add_argument('-learning_rate', type=float, help='Learning rate use for training the model', default=1e-5)
parser.add_argument('-epochs', type=int, help='number of training epochs', default=5)
parser.add_argument('-bert_frozen', default=False, action='store_true', help='flag to mark if the bert backbone if trainable or not')
parser.add_argument('-use_generator', default=False, action='store_true', help='flag to mark the usage of generators rather than loading all data in tf.data.Dataset')
parser.add_argument('-k', nargs='+', help='List of k\'s to use for evaluating metrics @k', default=[1, 3, 5])
parser.add_argument('-run_type', default="train", help='train or eval run')
parser.add_argument('-dev_run', default="True", help='whether it is a development run or production run (use all data, text external, less verbose etc)')
args = parser.parse_args()

args.dev_run = eval(args.dev_run)

if args.args_path != None:
    with open(args.args_path, 'r') as f:
        args.__dict__ = json.load(f)


if 'train' in args.run_type:

    if os.path.isdir(args.model_path):
        print("{0} already exists!".format(args.model_path))
        sys.exit()
    else:
        os.makedirs(args.model_path)

    with open(args.model_path +"/train_config.json", 'w') as f:
        json.dump(args.__dict__, f, indent = 2)

    eval_args = copy.deepcopy(args.__dict__)
    del eval_args['epochs']
    del eval_args['args_path']
    eval_args['run_type'] = "eval"
    eval_args['model_path'] = eval_args['model_path'] + "/weights/"

    with open(args.model_path +"/eval_config.json", 'w') as f:
        json.dump(eval_args, f, indent = 2)


def multiclass_rec(y, y_hat, top_k=None):
    m = tf.keras.metrics.Recall(top_k=top_k)
    m.update_state(y, y_hat)
    res = m.result().numpy()
    return res


def multiclass_prec(y, y_hat, top_k=None):
    m = tf.keras.metrics.Precision(top_k=top_k)
    m.update_state(y, y_hat)
    res = m.result().numpy()
    return res


def build_model(bert_model, number_of_labels):

    input_ids      = layers.Input(shape=(args.bert_max_seq_len,), dtype='int64', name="input_ids")
    attention_mask = layers.Input(shape=(args.bert_max_seq_len,), dtype='int32', name="attention_mask")

    bert_model.trainable = not args.bert_frozen
    bert_layer = bert_model(input_ids, attention_mask)[0]
    # get cls output
    bert_output = layers.Lambda(lambda seq: seq[:, 0, :])(bert_layer)
    # add dropout?
    classifier_layer = layers.Dense(number_of_labels, activation="sigmoid")(bert_output)

    model = keras.Model(inputs=[input_ids, attention_mask], outputs=classifier_layer)
    model.build(input_shape=[(None, args.bert_max_seq_len), (None, args.bert_max_seq_len)])
    if args.dev_run == True:
        model.summary()
    model.compile(tf.keras.optimizers.Adam(lr=args.learning_rate), loss='binary_crossentropy', metrics=[])

    return model


def build_dataset(inputs, labels, labels_dict, tokenizer):
    
    # build appropiate labels
    processed_labels = []
    for label in labels:
        onehot_labels = [0 for _ in range(len(labels_dict))]
        for l in label:
            onehot_labels[labels_dict[l]] = 1
        processed_labels.append(onehot_labels)

    def generator():
        for x in range(len(labels)):
            yield (inputs[0][x], inputs[1][x]), processed_labels[x]

    if args.use_generator == False:
        tf_dataset = tf.data.Dataset.from_tensor_slices(((inputs[0], inputs[1]), processed_labels))
    else:
        # build generator
        tf_dataset = tf.data.Dataset.from_generator(generator, output_types=((tf.int32, tf.int32), tf.int32),  output_shapes=(((args.bert_max_seq_len), (args.bert_max_seq_len)), (len(labels_dict))))
    return tf_dataset


def load_data_splits(inputs, labels):

    if os.path.exists(args.data_path+"_train_idx.pkl") and os.path.exists(args.data_path+"_dev_idx.pkl") and os.path.exists(args.data_path+"_test_idx.pkl"):
        train_indexes = pickle.load(open(args.data_path+"_train_idx.pkl", "rb"))
        dev_indexes = pickle.load(open(args.data_path+"_dev_idx.pkl", "rb"))
        test_indexes = pickle.load(open(args.data_path+"_test_idx.pkl", "rb"))
    
    else:
        # build indices: 80% train, 10% dev, 10% test
        indexes = set(range(len(inputs[0])))
        train_indexes = set(random.sample(indexes, int(0.8*len(indexes))))
        remaining_indexes = indexes.difference(train_indexes)
        dev_indexes = set(random.sample(remaining_indexes, int(0.1*len(indexes))))
        test_indexes = remaining_indexes.difference(dev_indexes)
        
        pickle.dump(train_indexes, open(args.data_path+"_train_idx.pkl", "wb"))
        pickle.dump(dev_indexes, open(args.data_path+"_dev_idx.pkl", "wb"))
        pickle.dump(test_indexes, open(args.data_path+"_test_idx.pkl", "wb"))

    if len(train_indexes.union(dev_indexes).union(test_indexes)) != len(inputs[0]) or len(train_indexes.intersection(dev_indexes).intersection(test_indexes)) != 0:
        print("Problem with indexes")
        print(len(train_indexes), len(dev_indexes), len(test_indexes))
        print("Union:", len(train_indexes.union(dev_indexes).union(test_indexes)), "should be:", len(inputs[0]))
        print("Intersection:", len(train_indexes.intersection(dev_indexes).intersection(test_indexes)))
        sys.exit()

    train_input_ids, train_input_attention, train_labels, dev_input_ids, dev_input_attention, dev_labels, test_input_ids, test_input_attention, test_labels = [], [], [], [], [], [], [], [], []

    for x in train_indexes:
        train_input_ids.append(inputs[0][x])
        train_input_attention.append(inputs[1][x])
        train_labels.append(labels[x])
    
    for x in dev_indexes:
        dev_input_ids.append(inputs[0][x])
        dev_input_attention.append(inputs[1][x])
        dev_labels.append(labels[x])
    
    for x in test_indexes:
        test_input_ids.append(inputs[0][x])
        test_input_attention.append(inputs[1][x])
        test_labels.append(labels[x])
    
    return [train_input_ids, train_input_attention], train_labels, [dev_input_ids, dev_input_attention], dev_labels, [test_input_ids, test_input_attention], test_labels


def load_data(tokenizer, load_external = False):
    # load external data for prod run
    if load_external == True:
        if "tags_titles" in args.data_path:
            task = "qa"
        elif "tags" in args.data_path:
            task = "tags"
            
        documents = pickle.load(open("_cache/_data/test_corpus_{0}".format(task) + "_x_data.pkl", "rb"))
        inputs = tokenizer(documents, padding="max_length", truncation=True, max_length=args.bert_max_seq_len, is_split_into_words=True)
        inputs = [inputs["input_ids"], inputs["attention_mask"]]
        labels = pickle.load(open("_cache/_data/test_corpus_{0}".format(task) + "_y_data.pkl", "rb"))
        labels_dict = pickle.load(open("_cache/_data/test_corpus_{0}".format(task) + "_labels_dict.pkl", "rb"))
        input_ids = []
        input_attention = []
        for x in range(len(documents)):
            input_ids.append(inputs[0][x])
            input_attention.append(inputs[1][x])
        return inputs, labels, None, None, None, None, labels_dict

    # we don't have external, we load data as usual (for dev run)
    # check if inputs are pre-processed (i.e. passed through tokenizer) for BERT-based model
    if not os.path.exists(args.data_path + "_x_data_inputs_len{0}.pkl".format(args.bert_max_seq_len)):
        print("###################### Building inputs file for BERT tokenizer. This may take a while. ######################")
        documents = pickle.load(open(args.data_path + "_x_data.pkl", "rb"))
        inputs = tokenizer(documents, padding="max_length", truncation=True, max_length=args.bert_max_seq_len, is_split_into_words=True)
        pickle.dump([inputs["input_ids"], inputs["attention_mask"]], open(args.data_path+"_x_data_inputs_len{0}.pkl".format(args.bert_max_seq_len), "wb"))
        inputs = [inputs["input_ids"], inputs["attention_mask"]]
    else:
        inputs = pickle.load(open(args.data_path + "_x_data_inputs_len{0}.pkl".format(args.bert_max_seq_len), "rb"))

    labels = pickle.load(open(args.data_path + "_y_data.pkl", "rb"))
    labels_dict = pickle.load(open(args.data_path + "_labels_dict.pkl", "rb"))

    train_documents, train_labels, dev_documents, dev_labels, test_documents, test_labels = load_data_splits(inputs, labels)

    return train_documents, train_labels, dev_documents, dev_labels, test_documents, test_labels, labels_dict

class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, data):
        self.data = data
        self.y_true = []
        for x in self.data:
            self.y_true.extend(x[1])
        self.y_true = np.array(self.y_true)

        if args.k == None:
            self.ks = [None]
        else:
            self.ks = list(map(lambda x: int(x), args.k))        
        if None not in self.ks and args.dev_run == True:
            self.ks.append(None)
            
        
    def on_epoch_end(self, epoch, logs=None):

        y_pred = self.model.predict(self.data)
        sorted_zip = [sorted(zip(y_pred[i], self.y_true[i]), reverse=True) for i in range(len(y_pred))]
        sorted_y_pred = np.array([list(map(lambda x: x[0], z)) for z in sorted_zip])
        sorted_y_true = np.array([list(map(lambda x: x[1], z)) for z in sorted_zip])
        print(flush=True)
        for k in self.ks:
            # based on tf notes this seems wrong (see link below)
            # https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Precision#update_state
            # but, if y_pred are not all 1 and are actual preds (floats) it works as intended
            # we can also use sorted_y_true and sorted_y_pred with exactly the same results
            rec = multiclass_rec(self.y_true, y_pred, top_k = k)
            prec = multiclass_prec(self.y_true, y_pred, top_k = k)

            correct = 0
            total = 0 

            ones_g = 0

            if k == None:
                k = len(y_pred[0])

            for i in range(len(y_pred)):
                binary_preds = (sorted_y_pred[i][:k] > 0.5).astype(np.uint8)
                correct += (binary_preds==sorted_y_true[i][:k]).sum()
                total += k
                
                for index in range(k):
                    g = sorted_y_true[i][index]
                    if g == 1:
                        ones_g += 1

            f1 = 2 * prec * rec / (prec + rec)
            acc = 1.0 * correct / total
            
            if args.dev_run == False:
                print("We have a total of {0} test documents. For each document we make {1} predictions.".format(len(self.y_true), k))
                print("Relevant predictions {0}/{1} or {2:.2f}%".format(ones_g, len(self.y_true)*k, 100.0*ones_g/(len(self.y_true)*k)))
            else:
                print("Recall: {0:.4f} Precision: {1:.4f} F1: {2:.4f} Acc: {3:.4f} @{4}".format(rec, prec, f1, acc, k))
            
            
            rec = round(rec, 4)
            prec = round(prec, 4)
            f1 = round(f1, 4)
            acc = round(acc, 4)

            logs["recall@{0}".format(k)] = rec
            logs["precision@{0}".format(k)] = prec
            logs["f1@{0}".format(k)] = f1
            logs["accuracy@{0}".format(k)] = acc

        return logs


if __name__ == "__main__":
    if GPU_INDEX_FOR_TRAINING != -1:
        str_device = '/device:GPU:{}'.format(GPU_INDEX_FOR_TRAINING)
    else:
        str_device = '/device:CPU:0'

    with tf.device(str_device):

        bert_model = TFBertModel.from_pretrained(args.bert_backbone)
        tokenizer = AutoTokenizer.from_pretrained(args.bert_backbone, use_fast=False)

        train_inputs, train_labels, dev_inputs, dev_labels, test_inputs, test_labels, labels_dict = load_data(tokenizer)
        print(len(train_labels), len(dev_labels), len(test_labels))
        if 'train' in args.run_type:
            if args.run_type == 'train_dev':
                train_inputs[0].extend(dev_inputs[0])
                train_inputs[1].extend(dev_inputs[1])
                train_labels.extend(dev_labels)
            
            if args.run_type == 'train_full':
                train_inputs[0].extend(dev_inputs[0])
                train_inputs[1].extend(dev_inputs[1])
                train_labels.extend(dev_labels)
                train_inputs[0].extend(test_inputs[0])
                train_inputs[1].extend(test_inputs[1])
                train_labels.extend(test_labels)

            train_dataset = build_dataset(train_inputs, train_labels, labels_dict, tokenizer).shuffle(40000).batch(args.batch_size)
        dev_dataset = build_dataset(dev_inputs, dev_labels, labels_dict, tokenizer).batch(args.batch_size)
        if args.run_type == 'eval' or args.dev_run == False:
            test_dataset = build_dataset(test_inputs, test_labels, labels_dict, tokenizer).batch(args.batch_size)
            test_callback = MetricsCallback(test_dataset)
        
        if args.dev_run == False:
            external_inputs, external_labels, _, _, _, _, external_labels_dict = load_data(tokenizer, load_external = True)
            print(len(external_labels), len(external_labels_dict), len(external_inputs[0]))
            external_test_dataset = build_dataset(external_inputs, external_labels, external_labels_dict, tokenizer).batch(1)
            external_test_callback = MetricsCallback(external_test_dataset)


        dev_callback = MetricsCallback(dev_dataset)
        model = build_model(bert_model, len(labels_dict))

        if 'train' in args.run_type:
            if ("_dev" in args.run_type or "_full" in args.run_type) and args.dev_run == True:
                print("###################### Reminder: scores are computed on the dev set that is part of training for this particular instance! ######################")
            

            if args.dev_run == True:
                checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=args.model_path+"/weights/{epoch:02d}", save_weights_only = True)
                history = model.fit(train_dataset, epochs=args.epochs, callbacks=[dev_callback, checkpoint_callback])
            else:
                history = model.fit(train_dataset, epochs=args.epochs, callbacks=[])
                model.save_weights(args.model_path+"/weights/{:02d}".format(args.epochs))
                external_test_callback.model = model
                external_test_results = external_test_callback.on_epoch_end(0, {})
                    
                
            hist_df = pd.DataFrame(history.history)
            hist_json_file = '{0}/history.json'.format(args.model_path)
            with open(hist_json_file, mode='w') as f:
                hist_df.to_json(f, indent=2)


        elif args.run_type == 'eval':
            model.load_weights(args.model_path)

            print("Dev results:", end="")
            dev_callback.model = model
            dev_results = dev_callback.on_epoch_end(0, {})
            print(dev_results)
            print()

            print("Test results:",end="")
            test_callback.model = model
            test_results = test_callback.on_epoch_end(0, {})
            print(test_results)
            
            # save to hf format
            # path_parts = args.model_path.split("weights")
            # save_path = os.path.join(path_parts[0], path_parts[1][1:])
            # bert_model.save_pretrained(save_path)
            # tokenizer.save_pretrained(save_path)

