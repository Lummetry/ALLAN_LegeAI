import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)
import pickle
from transformers import TFBertForSequenceClassification, TFAutoModel, AutoTokenizer, TFBertModel
import numpy as np
from tensorflow.keras import layers
from tensorflow import keras
import sys
import random
import os
import time
from sklearn.metrics import classification_report

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 5

DATA_PATH = "_cache/_data/20220209_211238"

USE_GENERATOR = True

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

    input_ids      = layers.Input(shape=(MAX_LENGTH,), dtype='int64', name="input_ids")
    attention_mask = layers.Input(shape=(MAX_LENGTH,), dtype='int32', name="attention_mask")

    # bert_model.trainable = False
    bert_layer = bert_model([input_ids, attention_mask])[0]
    # get cls output
    bert_output = layers.Lambda(lambda seq: seq[:, 0, :])(bert_layer)
    # add dropout?
    classifier_layer = layers.Dense(number_of_labels, activation="sigmoid")(bert_output)

    model = keras.Model(inputs=[input_ids, attention_mask], outputs=classifier_layer)
    model.build(input_shape=[(None, MAX_LENGTH), (None, MAX_LENGTH), (None, MAX_LENGTH)])

    model.summary()

    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='binary_crossentropy', metrics=[])

    return model


def build_dataset(inputs, labels, labels_dict, tokenizer):
    
    # build appropiate labels
    processed_labels = []
    for label in labels:
        onehot_labels = [0 for _ in range(len(labels_dict))]
        for l in label:
            onehot_labels[labels_dict[l]] = 1
        processed_labels.append(onehot_labels)

    # inputs = tokenizer(documents, padding="max_length", truncation=True, max_length=MAX_LENGTH, is_split_into_words=True)

    def generator():
        for x in range(len(labels)):
            yield (inputs[0][x], inputs[1][x]), processed_labels[x]

    if USE_GENERATOR == False:
        tf_dataset = tf.data.Dataset.from_tensor_slices(((inputs[0], inputs[1]), processed_labels))
    else:
        # build generator
        tf_dataset = tf.data.Dataset.from_generator(generator, output_types=((tf.int32, tf.int32), tf.int32),  output_shapes=(((MAX_LENGTH), (MAX_LENGTH)), (len(labels_dict))))
    return tf_dataset


def load_data_splits(inputs, labels):

    if os.path.exists(DATA_PATH+"_train_idx.pkl") and os.path.exists(DATA_PATH+"_dev_idx.pkl") and os.path.exists(DATA_PATH+"_test_idx.pkl"):
        train_indexes = pickle.load(open(DATA_PATH+"_train_idx.pkl", "rb"))
        dev_indexes = pickle.load(open(DATA_PATH+"_dev_idx.pkl", "rb"))
        test_indexes = pickle.load(open(DATA_PATH+"_test_idx.pkl", "rb"))
    
    else:
        # build indices: 80% train, 10% dev, 10% test
        indexes = set(range(len(inputs[0])))
        train_indexes = set(random.sample(indexes, int(0.8*len(indexes))))
        remaining_indexes = indexes.difference(train_indexes)
        dev_indexes = set(random.sample(remaining_indexes, int(0.1*len(indexes))))
        test_indexes = remaining_indexes.difference(dev_indexes)
        
        pickle.dump(train_indexes, open(DATA_PATH+"_train_idx.pkl", "wb"))
        pickle.dump(dev_indexes, open(DATA_PATH+"_dev_idx.pkl", "wb"))
        pickle.dump(test_indexes, open(DATA_PATH+"_test_idx.pkl", "wb"))

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
    if not os.path.exists(DATA_PATH + "_x_data_inputs_len{0}.pkl".format(MAX_LENGTH)):
        print("Building inputs file. This may take a while.")
        documents = pickle.load(open(DATA_PATH + "_x_data.pkl", "rb"))
        inputs = tokenizer(documents, padding="max_length", truncation=True, max_length=MAX_LENGTH, is_split_into_words=True)
        pickle.dump([inputs["input_ids"], inputs["attention_mask"]], open(DATA_PATH+"_x_data_inputs_len{0}.pkl".format(MAX_LENGTH), "wb"))
        inputs = [inputs["input_ids"], inputs["attention_mask"]]
    else:
        inputs = pickle.load(open(DATA_PATH + "_x_data_inputs_len{0}.pkl".format(MAX_LENGTH), "rb"))
        

    labels = pickle.load(open(DATA_PATH + "_y_data.pkl", "rb"))
    labels_dict = pickle.load(open(DATA_PATH + "_labels_dict.pkl", "rb"))

    train_documents, train_labels, dev_documents, dev_labels, test_documents, test_labels = load_data_splits(inputs, labels)

    return train_documents, train_labels, dev_documents, dev_labels, test_documents, test_labels, labels_dict

class MetricsCallback(keras.callbacks.Callback):
    def __init__(self, data):
        self.data = data
        self.y_true = []
        for x in self.data:
            self.y_true.extend(x[1])
        self.y_true = np.array(self.y_true)
        
    def on_epoch_end(self, epoch, logs=None):

        y_pred = self.model.predict(self.data)
        print(flush=True)
        ks = [1,3,5,None]

        sorted_zip = [sorted(zip(y_pred[i], self.y_true[i]), reverse=True) for i in range(len(y_pred))]
        sorted_y_pred = np.array([list(map(lambda x: x[0], z)) for z in sorted_zip])
        sorted_y_true = np.array([list(map(lambda x: x[1], z)) for z in sorted_zip])

        for k in ks:
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





if __name__ == "__main__":

    st = time.time()
    bert_model = TFBertModel.from_pretrained("readerbench/jurBERT-base")
    tokenizer = AutoTokenizer.from_pretrained("readerbench/jurBERT-base")

    train_inputs, train_labels, dev_inputs, dev_labels, test_inputs, test_labels, labels_dict = load_data(tokenizer)

    train_dataset = build_dataset(train_inputs, train_labels, labels_dict, tokenizer).shuffle(10000).batch(BATCH_SIZE)
    dev_dataset = build_dataset(dev_inputs, dev_labels, labels_dict, tokenizer).batch(BATCH_SIZE)
    test_dataset = build_dataset(test_inputs, test_labels, labels_dict, tokenizer).batch(BATCH_SIZE)

    dev_callback = MetricsCallback(dev_dataset)

    model = build_model(bert_model, len(labels_dict))

    model.fit(train_dataset, epochs=EPOCHS, callbacks=[dev_callback])
    print(time.time()-st)
