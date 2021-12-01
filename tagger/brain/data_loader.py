# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 08:13:19 2019

@author: Andrei DAMIAN


"""
import tensorflow as tf
import numpy as np
from tagger.brain.base_engine import ALLANTaggerEngine

class ALLANDataLoader(ALLANTaggerEngine):
  
  def __init__(self, 
               multi_label=True, 
               normalize_labels=False, 
               **kwargs):
    super().__init__(**kwargs)
    self.__name__ = 'AT_DL'
    self.multi_label = multi_label
    self.normalize_labels  = normalize_labels
    self._setup()
    return
  
  def _setup(self):
    self.train_subfolders_config = self.train_config['SUBFOLDERS'] if 'SUBFOLDERS' in self.train_config.keys() else None
    if self.train_subfolders_config is not None:
      self.train_subfolders = self.train_subfolders_config['ENABLED']
      self.docs_subfolder = self.train_subfolders_config['DOCS']
      self.labels_subfolder  = self.train_subfolders_config['LABELS']
    else:
      self.train_subfolders = False
    if not self.train_subfolders:
      self.docs_subfolder = None
      self.labels_subfolder  = None
    return
     
  
  def LoadData(self):
    fn_w2i = self.fn_word2idx
    fn_i2w = self.fn_idx2word
    fn_l2i = self.fn_labels2idx
    fld_train = self.train_folder
    fld_docs = self.docs_subfolder
    fld_labs = self.labels_subfolder
    self.x_docs, self.y_labels = self._load_training_data(
                                    train_folder=fld_train,
                                    doc_subfolder=fld_docs,
                                    lab_subfolder=fld_labs,
                                    doc_ext=self.doc_ext,
                                    label_ext=self.label_ext,
                                    fn_words_dict=fn_w2i,
                                    fn_idx_dict=fn_i2w,
                                    fn_labels_dict=fn_l2i,
                                    multi_label=self.multi_label,
                                    normalize=self.normalize_labels
                                    )
    return
  
  
    
    
  
  def _load_training_data(self, train_folder, 
                          doc_subfolder, lab_subfolder,                          
                          doc_ext, label_ext,
                          fn_words_dict, 
                          fn_idx_dict,
                          fn_labels_dict,
                          save_dicts=True, 
                          multi_label=True, normalize=False):
    """
     utility function to load training data and tokenize as follows:
     if word2idx is none then use tf tokenizer and save dict
    """

    dict_labels2idx = None
    self.P("Loading labels file '{}'".format(fn_labels_dict))
    if fn_labels_dict is not None:
      if ".txt" in fn_labels_dict:
        dict_labels2idx = self.log.LoadDictFromData(fn_labels_dict)
      else:
        dict_labels2idx = self.log.LoadPickleFromData(fn_labels_dict)
    if dict_labels2idx is None:
      self.P(" No labels2idx dict found")
    
    self._setup_vocabs(fn_words_dict, fn_idx_dict)
    dict_word2idx, dict_idx2word = self.dic_word2index, self.dic_index2word
    
    _res = self.log.LoadDocuments(train_folder,
                                  doc_ext=doc_ext,
                                  label_ext=label_ext,
                                  doc_folder=doc_subfolder,
                                  label_folder=lab_subfolder)
    lst_docs, lst_labels, lst_unique_lab = _res
    no_labels = lst_labels.count(None)
    if no_labels:
      raise ValueError("Found {} document without labels!".format(no_labels))

    self.raw_documents = lst_docs
    self.raw_labels = lst_labels


    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(lst_docs)
    self.lst_loaded_words = [k for k in tokenizer.word_index]

    if dict_word2idx is None:
      x_docs = tokenizer.texts_to_sequences(lst_docs)
      self.dic_word2index = tokenizer.word_index
      self.tokenizer = tokenizer
      self.P("Generated {} words word2idx dict: {}".format(
          len(self.dic_word2index), 
          ["{}:{}".format(k,v) for k,v in self.dic_word2index.items()][:4]))
      if save_dicts:
        self.log.SaveDataJSON(self.dic_word2index,'auto_word2idx.txt')
    else:
      self.P("Using predefined word2idx with {} words: {}".format(
          len(dict_word2idx), 
          ["{}:{}".format(k,v) for k,v in dict_word2idx.items()][:4]))
      lst_splitted = [self._get_words(x) for x in lst_docs]
      x_docs = []
      for text in lst_splitted:
        x_docs.append([self._word_encoder(x) for x in text])
        
    lens = [len(d) for d in x_docs]
    min_len = min(lens)
    max_len = max(lens)
    avg_len = np.mean(lens)
    med_len = np.median(lens)
    
    if dict_labels2idx is None:
      dict_labels2idx = {k:v for v,k in enumerate(np.unique(lst_unique_lab))}
      if save_dicts:
        self.log.SaveDataJSON(dict_labels2idx, 'auto_labels2idx.txt')

    self.dic_labels = dict_labels2idx
   
    self.output_size = len(dict_labels2idx)
    y_labels = []
    tags = []
    for doc in lst_labels:
      doc_labels = [dict_labels2idx[x] for x in doc]
      if multi_label:
        doc_labels_one_hot = tf.keras.utils.to_categorical(doc_labels,
                                                           num_classes=self.output_size)
        label = doc_labels_one_hot.sum(axis=0).astype(int).tolist()
        if normalize:
          label = np.array(label) / sum(label)
        tags.append(sum(label))
      else:
        label = doc_labels[0]
        tags.append(1)
      y_labels.append(label)
    if dict_idx2word is None:
      dict_idx2word = {v:k for k,v in self.dic_word2index.items()}
    self.dic_index2word = dict_idx2word
    self.dic_index2labels = {v:k for k,v in self.dic_labels.items()}
    self.P("Loaded {} docs w. {} total tags, max {} tags/obs".format(
              len(x_docs), self.output_size, np.max(tags)))
    self.P("  Min doc word len: {}".format(min_len))
    self.P("  Max doc word len: {}".format(max_len))
    self.P("  Avg doc word len: {}".format(avg_len))
    self.P("  Med doc word len: {}".format(med_len))
    self.log.ShowTextHistogram(lens, caption='Doc word len distrib', )
    self.P("  Loaded documents vocabulary: {}".format(len(self.lst_loaded_words)))
    self.P("  Words-to-indexes vocabulary: {}".format(len(self.dic_word2index)))
    return x_docs, y_labels
  
    

if __name__ == '__main__':
  cfg1 = "tagger/brain/config.txt"
  
  from libraries import Logger


  l = Logger(lib_name="ALNT",config_file=cfg1)
  eng = ALLANDataLoader(log=l, multi_label=True, normalize_labels=True)
  eng.LoadData()
  
  txt_test = [
      "Ana are mere si maria are pere",
      "Asta este e chestie cool si trendy"
      ]
  lbl_test = [
      ['gastro', 'orl'],
      ['dieta']
      ]
  et1,el1 = eng.encode(txt_test, lbl_test, to_onehot=True)
  print(et1, el1)
  
  t1,tl1 = eng.decode(et1, el1, from_onehot=True) 
  print(t1, tl1)
  
    