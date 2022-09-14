from endpoints.get_tags_orig import GetTagsWorkerOrig
from endpoints.get_tags import GetTagsWorker
from libraries import Logger
import itertools
import pickle
import numpy as np
import os
import csv
import sys
import copy
# import Levenshtein as lev


import tensorflow as tf


# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

_CONFIG_V1 = {
  'TAGGER_MODEL': '20211206_205159_ep35_R0.61_P0.90_F10.73.h5',
  'LABEL2ID': 'dict_lbl_37.pkl',
  'EMBGEN_MODEL': '20211125_203842_embgen_model_sc_40_ep140.h5',
  'GENERATED_EMBEDS': 'embgen_full_embeds.npy',
  'WORD_EMBEDS': 'lai_embeddings_191K.pkl',
  'IDX2WORD': 'lai_ro_i2w_191K.pkl'

}

_CONFIG_V2 = {
  "TAGGER_MODEL" : "_cache/_models/tags_5/weights/05",
  "LABEL2ID" : "tags_v1_labels_dict.pkl",
  "BERT_BACKBONE" : "readerbench/jurBERT-base",
  "BERT_MAX_SEQ_LEN" : 512
}


data_path = "_cache/_data/tags_v1"
split = "test"


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


def stat():

    tags = pickle.load(open("_cache/_data/tags_titles_v1_labels_dict.pkl", "rb"))
    # qav1 = pickle.load(open("_cache/_data/qa_v1_labels_dict.pkl", "rb"))
    # qav2 = pickle.load(open("_cache/_data/qa_v2_labels_dict.pkl", "rb"))

    # print(len(tags), len(qav1), len(qav2))
    print(len(tags))
    

    tags = set(tags.keys())
    # qav1 = set(qav1.keys())
    # qav2 = set(qav2.keys())
    
    # print(len(tags), len(qav1), len(qav2))

    # qav1-qav2
    # print("#"*50)
    # print("QAV1-QAV2")
    # i1 = qav1.intersection(qav2)
    # i2 = qav2.intersection(qav1)
    # print(len(i1), len(i2), i1==i2)

    # d1 = qav1.difference(qav2)
    # d2 = qav2.difference(qav1)
    # print(len(d1), len(d2))
    # print("In v2 not in v1:", d2)

    # print("#"*50)
    
    # tags-qav2
    # print("TAGS-QAV2")
    # i1 = tags.intersection(qav2)
    # i2 = qav2.intersection(tags)
    # print(len(i1), len(i2), i1==i2)
    # print("Both in tags and qav2:", i1)
    
    # d1 = tags.difference(qav2)
    # d2 = qav2.difference(tags)
    # print(len(d1), len(d2))

    # print("#"*50)
    # f = []
    # print("QAV2-QAV2")
    # for i in itertools.product(list(qav2), list(qav2)):
    #     lev_d = lev.distance(i[0], i[1])
    #     lev_r = lev.ratio(i[0], i[1])
    #     if (lev_d < 3 or lev_r > 0.75) and lev_d > 0 and [i[0], i[1]] not in f:
    #         print(i, lev_d, lev_r)
    #         f.append([i[0], i[1]])
    #         f.append([i[1], i[0]])

    # print("#"*50)
    f = []
    print("TAGS-TAGS")
    for i in itertools.product(list(tags), list(tags)):
        lev_d = lev.distance(i[0], i[1])
        lev_r = lev.ratio(i[0], i[1])
        if (lev_d < 3 or lev_r > 0.85) and lev_d > 0 and [i[0], i[1]] not in f:
            print(i, lev_d, lev_r)
            f.append([i[0], i[1]])
            f.append([i[1], i[0]])


    print("#"*50)
    # print("TAGS-QAV2")
    # for i in itertools.product(list(tags), list(qav2)):
    #     lev_d = lev.distance(i[0], i[1])
    #     lev_r = lev.ratio(i[0], i[1])
    #     if (lev_d < 3 or lev_r > 0.75) and lev_d > 0:
    #         print(i, lev_d, lev_r)
    print()


def preprocess_titles():

    REMOVE_PARAN = 0
    REMOVE_PREFIX = 1
    REMOVE_POS = 2
    REMOVE_STOPWORDS = 3
    REMOVE_DEP = 4
    REMOVE_NONALPHA = 5
    REMOVE_ENTITIES = 6


    # titles = pickle.load(open("_cache/_data/tags_titles_v1_x_data.pkl", "rb"))
    # tags   = pickle.load(open("_cache/_data/tags_titles_v1_y_data.pkl", "rb"))

    titles = pickle.load(open("_cache/_data/20220405_123946_x_data.pkl", "rb"))
    tags   = pickle.load(open("_cache/_data/20220405_123946_y_data.pkl", "rb"))
    print(len(titles), len(tags))

    import spacy
    from utils.utils import preprocess_title
    nlp = spacy.load('ro_core_news_lg')

    import random
    # idx = random.sample(range(0, len(titles)), 3)
    idx = list(range(len(tags)))
    t = 0

    new_titles = []
    new_tags = []

    for i in idx:
        title = " ".join(titles[i])
        # print("original:", title)
        res = preprocess_title(title, nlp=nlp, proc=[REMOVE_PARAN, REMOVE_PREFIX, REMOVE_POS, REMOVE_DEP, REMOVE_NONALPHA, REMOVE_ENTITIES])
        # print("processed:", res)
        if (i+1) % 5000 == 0:
            print(i+1, t)

        
        if len(res.split(" ")) <= 2 or len(res.split(" ")) > 20:
            continue
        
        new_titles.append(titles[i])
        new_tags.append(tags[i])

    

        t += 1

    print(t)
    print(len(new_titles))
    print(len(new_tags))

    pickle.dump(new_titles, open("_cache/_data/tags_titles_v1_x_data.pkl", "wb"))
    pickle.dump(new_tags, open("_cache/_data/tags_titles_v1_y_data.pkl", "wb"))

    
def get_embed():
    from libraries import Logger
    from tagger.brain.emb_aproximator import SimpleEmbeddingApproximatorWrapper
    import scipy.spatial.distance as distance



    l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)

    encoder = SimpleEmbeddingApproximatorWrapper(
          log=l,
          fn_embeds="_cache/_data/lai_embeddings_191K.pkl",
          fn_idx2word="lai_ro_i2w_191K.pkl",
          embgen_model_file="20211125_203842_embgen_model_sc_40_ep140.h5",
          generated_embeds_filename="embgen_full_embeds.npy",
        )

    dist_func = encoder.encoder._setup_dist_func(func_name='cos')


    tags_labels = pickle.load(open("_cache/_data/tags_titles_v1_labels_dict.pkl", "rb"))
    labels_tags = []
    for k,v in tags_labels.items():
        if k not in labels_tags:
            labels_tags.append(k)
    
    labels_qa = []
    with open("_cache/_data/qa_labels_from_db.csv") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row_index, row in enumerate(csv_reader):
            if row_index == 0:
                continue
            if "ANEXA" in row[1] or "0 5" in row[1]:
                continue
            labels_qa.append(row[1])

    TYPE = "split"

    labels_tags_embs = []
    for label_tag in labels_tags:
        if TYPE == "single":
            tag_embed = encoder.encode_convert_unknown_words([[label_tag]], fixed_len=0)[0] 
            tag_embed = np.squeeze(np.array(tag_embed))
        elif TYPE == "split":
            label_tag = label_tag.split("_")[0:1]
            tag_embed = encoder.encode_convert_unknown_words([label_tag], fixed_len=0)[0]         
            tag_embed = np.mean(np.array(tag_embed), axis=0)

        labels_tags_embs.append(tag_embed)

    
    labels_qa_embs = []
    for label_qa in labels_qa:
        # label_qa = label_qa.lower()
        if TYPE == "single":
            label_qa = "_".join(label_qa.split(" "))
            qa_embed = encoder.encode_convert_unknown_words([[label_qa]], fixed_len=0)[0] 
            qa_embed = np.squeeze(np.array(qa_embed))

        elif TYPE == "split":
            label_qa = label_qa.split(" ")[0:1]
            qa_embed = encoder.encode_convert_unknown_words([label_qa], fixed_len=0)[0] 
            qa_embed = np.mean(np.array(qa_embed), axis=0)

        labels_qa_embs.append(qa_embed)

    print(len(labels_tags), np.array(labels_tags_embs).shape)
    print(len(labels_qa), np.array(labels_qa_embs).shape)

    data = []

    for tag_index in range(len(labels_tags)):
        max_sim = 0
        max_index = None
        sims = []
        for qa_index in range(len(labels_qa)):
            sim = 1 - distance.cosine(labels_tags_embs[tag_index], labels_qa_embs[qa_index])
            if sim > max_sim:
                max_sim = sim
                max_index = qa_index
            sims.append([qa_index, sim])
        sims.sort(key=lambda x: x[1], reverse=True)
        # print(labels_tags[tag_index], "-", list(map(lambda x: [labels_qa[x[0]], x[1]], sims[:5])))
        # print()
        entry = [labels_tags[tag_index], ", ".join(list(map(lambda x: labels_qa[x[0]], sims[:5])))]
        data.append(entry)

    # print(data[-1])
    with open('tags_qa.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["TAG", "Top5 QA"])
        for entry in data:
            writer.writerow(entry)

    sys.exit()


def split_csv(file):
    import csv
    import ctypes as ct

    csv.field_size_limit(int(ct.c_ulong(-1).value // 2))
    row_count = 0
    pr = True
    rows_to_save = []
    saved_rows = 0
    with open(file, encoding="utf-8") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row_index, row in enumerate(csv_reader):
            pr = True
            if row_index == 0:
                continue            
            if row != []:
                row_count += 1
                pr = False
                row[1] = row[1].replace("\n<NL>", "<NL>")
                row[3] = row[3].replace("\n<NL>", "<NL>")
                row[5] = row[5].replace("\n<NL>", "<NL>")
                rows_to_save.append(row)
            if row_count != 0 and row_count % 100000 == 0 and pr:
                print(row_count)
                with open("_cache/_data/merge/merge{0}.csv".format((row_count+1)//100000), "w", encoding="utf-8") as csv_file_out:
                    writer = csv.writer(csv_file_out)
                    writer.writerow(["ID_Pasiv", "Pasiv", "ID_Activ", "Activ", "ID_Transformat", "Transformat"])
                    for row_to_save in rows_to_save:
                        writer.writerow(row_to_save)
                        saved_rows += 1
                    rows_to_save = []
                pr = True




    print(row_count)

    with open("_cache/_data/merge/merge{0}.csv".format((row_count+1)//100000), "w", encoding="utf-8") as csv_file_out:
        writer = csv.writer(csv_file_out)
        writer.writerow(["ID_Pasiv", "Pasiv", "ID_Activ", "Activ", "ID_Transformat", "Transformat"])
        for row_to_save in rows_to_save:
            writer.writerow(row_to_save)
            saved_rows += 1
    print(saved_rows)


def recover_text_from_xdata(x_data_file):
    import random
    docs = pickle.load(open(x_data_file, "rb"))
    indexes = set(range(len(docs)))
    
    train_indexes =  set(random.sample(indexes, int(0.9*len(indexes))))
    print(len(indexes), len(train_indexes))
    with open("texts_train.txt", "w", encoding="utf-8") as train_file:
        with open("texts_test.txt", "w", encoding="utf-8") as test_file:
            for doc_index, doc in enumerate(docs):
                if doc_index in train_indexes:
                    train_file.write(" ".join(doc)+"\n\n")
                else:
                    test_file.write(" ".join(doc)+"\n\n")
            

def get_preds_scores(preds, gts):

    sorted_zip = [sorted(zip(preds[i], gts[i]), reverse=True) for i in range(len(preds))]
    sorted_y_pred = np.array([list(map(lambda x: x[0], z)) for z in sorted_zip])
    sorted_y_true = np.array([list(map(lambda x: x[1], z)) for z in sorted_zip])

    for k in [1, 3, 5]:
        rec = multiclass_rec(gts, preds, top_k = k)
        prec = multiclass_prec(gts, preds, top_k = k)

        correct = 0
        total = 0 

        if k == None:
            k = len(gts[0])

        for i in range(len(gts)):
            binary_preds = (sorted_y_pred[i][:k] > 0.5).astype(np.uint8)
            correct += (binary_preds==sorted_y_true[i][:k]).sum()
            total += k

        f1 = 2 * prec * rec / (prec + rec)
        acc = 1.0 * correct / total

        print("Recall: {0:.4f} Precision: {1:.4f} F1: {2:.4f} Acc: {3:.4f} @{4}".format(rec, prec, f1, acc, k))


def stats_pred():

    dev = pickle.load(open("preds_dev.pkl", "rb"))

    print(len(dev), len(dev[0]), len(dev[1]), len(dev[2]))
    bpreds = []
    opreds = []
    gts = []
    for i in range(len(dev[0])):
        bert_pred = dev[0][i][:3]
        orig_pred = dev[1][i][:3]
        gt = dev[2][i]

        bpreds.extend(list(map(lambda x: x[0], bert_pred)))
        opreds.extend(list(map(lambda x: x[0], orig_pred)))
        gts.extend(gt)
        
        # break

    print(len(bpreds), len(opreds), len(gts))
    # print(bpreds)
    # print(opreds)
    # print(gts)

    from collections import Counter
    c = Counter(gts)
    print("total labels", len(c))
    t = 0
    tg = 0
    tb = 0
    to = 0
    for value, count in c.items():
        if count < 3:
            if bpreds.count(value) > opreds.count(value): 
                print(value, count)
                print("b", bpreds.count(value), "o", opreds.count(value))
            tg += count
            tb += bpreds.count(value)
            to += opreds.count(value)
            t += 1
    print("total", t)
    print(tg, tb, to)


    labels = ["meteorologie"]

    for i in range(len(dev[0])):
        bert_pred = dev[0][i][:3]
        orig_pred = dev[1][i][:3]
        gt = dev[2][i]

        for label in labels:
            if label in gt:
                print(i)
                print(bert_pred)
                print(orig_pred)
                print(gt)
                print()



    # for x in dev:
    #     print(x[0])
    #     print()
    #     print(x[1])
    #     print()
    #     print(x[2])

if __name__ == "__main__":

    stats_pred()
    # sys.exit()
    # recover_text_from_xdata("_cache/_data/tags_v1_x_data.pkl")
    # sys.exit()
    # split_csv("_cache/_data/merge/merge_examples.csv")
    # sys.exit()
    # get_embed()
    # sys.exit()
    # stat()
    # preprocess_titles()
    # sys.exit()

    l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
    # worker_orig = GetTagsWorkerOrig(log=l, default_config=_CONFIG_V1, verbosity_level=0)
    # worker = GetTagsWorker(log=l, default_config=_CONFIG_V2, verbosity_level=0)

    test_indexes = pickle.load(open(data_path+"_{0}_idx.pkl".format(split), "rb"))
    inputs = pickle.load(open(data_path + "_x_data.pkl", "rb"))
    labels = pickle.load(open(data_path + "_y_data.pkl", "rb"))

    # processed_labels = []
    # for label in labels:
    #     onehot_labels = [0 for _ in range(len(worker.label_to_id))]
    #     for l in label:
    #         onehot_labels[worker.label_to_id[l]] = 1
    #     processed_labels.append(onehot_labels)

    # processed_labels_orig = []
    # for label in labels:
    #     onehot_labels = [0 for _ in range(len(worker_orig.label_to_id))]
    #     for l in label:
    #         onehot_labels[worker_orig.label_to_id[l]] = 1
    #     processed_labels_orig.append(onehot_labels)



    test_documents = []
    test_labels = []
    test_labels_orig = []

    test_indexes = list(test_indexes)


    start_index = 0
    end_index = 9999999999999
    
    #print(test_indexes[start_index], test_indexes[end_index])
    for c, x in enumerate(test_indexes):
        if c < start_index:
            continue
        
        test_documents.append(' '.join(inputs[x]))
        # test_labels.append(processed_labels[x])
        # test_labels_orig.append(processed_labels_orig[x])
        c += 1
        
        if c == end_index:
            break
           
    #print("next start_index", end_index, "value", test_indexes[end_index])
    print(test_documents[4588])
    sys.exit()
    del inputs
    print(len(test_documents))
    preds = []
    preds_orig = []

    ground_truth = []
    text_preds = []
    text_preds_orig = []
    for el in range(len(test_documents)):
        if el % 500 == 0:
            print(el)
        
        if el != 0 and el % 3000 == 0:
            pickle.dump([text_preds, text_preds_orig, ground_truth], open("preds_{0}_{1}.pkl".format(split, el), "wb"))
            gts = test_labels[:el]
            gts_orig = test_labels_orig[:el]
            print(len(test_documents), len(preds), len(preds_orig), len(gts), len(gts_orig), len(ground_truth), len(text_preds), len(text_preds_orig))

            get_preds_scores(preds, gts)
            print()
            get_preds_scores(preds_orig, gts_orig)
            
        doc = test_documents[el]

        ins = worker._pre_process({"DOCUMENT": doc, "TOP_N": 5})
        p = worker._predict(ins)
        r = worker._post_process(p)
        aux = []
        for ix, x in enumerate(test_labels[el]):
            if x == 1:
                aux.append(worker.id_to_label[ix])
        ground_truth.append(aux)
        preds.append(np.array(np.squeeze(p[0])))
        text_preds.append(r['results'])
        
        ins = worker_orig._pre_process({"DOCUMENT": doc, "TOP_N": 5})
        p = worker_orig._predict(ins)
        r = worker_orig._post_process(p)       
        preds_orig.append(np.array(np.squeeze(p[0])))
        text_preds_orig.append(r['results'])
        
    print(np.array(preds).shape, np.array(test_labels).shape)
    print(np.array(preds_orig).shape, np.array(test_labels_orig).shape)
    print(len(ground_truth))
    
    
    
    #x = np.array([preds, preds_orig])
    #print(type(x), x.shape)
    #np.save(open("preds_{0}.npy".format(split), "wb"), x)
    pickle.dump([text_preds, text_preds_orig, ground_truth], open("preds_{0}.pkl".format(split), "wb"))

    gts = test_labels
    gts_orig = test_labels_orig
    print(len(test_documents), len(preds), len(preds_orig), len(gts), len(gts_orig), len(ground_truth), len(text_preds), len(text_preds_orig))

    get_preds_scores(preds, gts)
    print()
    get_preds_scores(preds_orig, gts_orig)
    
