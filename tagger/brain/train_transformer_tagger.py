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


parser = argparse.ArgumentParser(description='Training and evaluation file for transformer-based document-level multilabel classifier')
parser.add_argument('-args_path', help='path to config file')
parser.add_argument('-data_path', help='path + files desciptor (i.e. _cache/_data/20220209_211238)', required=True)
parser.add_argument('-bert_max_seq_len', type=int, help='Bert maximum sequence length', required=True)
parser.add_argument('-batch_size', type=int, help='training batch size', required=True)
parser.add_argument('-learning_rate', type=float, help='Learning rate use for training the model', default=1e-5)
parser.add_argument('-epochs', type=int, help='number of training epochs', default=5)
parser.add_argument('-bert_frozen', default=False, action='store_true', help='flag to mark if the bert backbone if trainable or not')
parser.add_argument('-use_generator', default=False, action='store_true', help='flag to mark the usage of generators rather than loading all data in tf.data.Dataset')
parser.add_argument('-k', nargs='+', help='List of k\'s to use for evaluating metrics @k', default=[1, 3, 5])
args = parser.parse_args()

if args.args_path != None:
    with open(args.args_path, 'r') as f:
        args.__dict__ = json.load(f)


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


def load_data(tokenizer):
    # check if inputs are pre-processed (i.e. passed through tokenizer) for BERT-based model
    if not os.path.exists(args.data_path + "_x_data_inputs_len{0}.pkl".format(args.bert_max_seq_len)):
        print("Building inputs file. This may take a while.")
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
        if None not in self.ks:
            self.ks.append(None)
        
    def on_epoch_end(self, epoch, logs=None):

        y_pred = self.model.predict(self.data)
        sorted_zip = [sorted(zip(y_pred[i], self.y_true[i]), reverse=True) for i in range(len(y_pred))]
        sorted_y_pred = np.array([list(map(lambda x: x[0], z)) for z in sorted_zip])
        sorted_y_true = np.array([list(map(lambda x: x[1], z)) for z in sorted_zip])

        for k in self.ks:
            rec = multiclass_rec(self.y_true, y_pred, top_k = k)
            prec = multiclass_prec(self.y_true, y_pred, top_k = k)

            correct = 0
            total = 0 

            if k == None:
                k = len(y_pred[0])

            for i in range(len(y_pred)):
                binary_preds = (sorted_y_pred[i][:k] > 0.5).astype(np.uint8)
                correct += (binary_preds==sorted_y_true[i][:k]).sum()
                total += k

            f1 = 2 * prec * rec / (prec + rec)
            acc = 1.0 * correct / total
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

    bert_model = TFBertModel.from_pretrained("readerbench/jurBERT-base")
    tokenizer = AutoTokenizer.from_pretrained("readerbench/jurBERT-base")

    train_inputs, train_labels, dev_inputs, dev_labels, test_inputs, test_labels, labels_dict = load_data(tokenizer)

    train_dataset = build_dataset(train_inputs, train_labels, labels_dict, tokenizer).shuffle(10000).batch(args.batch_size)
    dev_dataset = build_dataset(dev_inputs, dev_labels, labels_dict, tokenizer).batch(args.batch_size)
    test_dataset = build_dataset(test_inputs, test_labels, labels_dict, tokenizer).batch(args.batch_size)

    dev_callback = MetricsCallback(dev_dataset)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath="logs/{epoch:02d}", save_weights_only = True)
    
    model = build_model(bert_model, len(labels_dict))
    history = model.fit(train_dataset, epochs=args.epochs, callbacks=[dev_callback, checkpoint_callback])

    hist_df = pd.DataFrame(history.history) 
    hist_json_file = 'logs/history.json' 
    with open(hist_json_file, mode='w') as f:
        hist_df.to_json(f, indent=2)

    # this is the evaluation
    # r = dev_callback.on_epoch_end(0, {})
    # print(r)
