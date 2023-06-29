# -*- coding: utf-8 -*-
"""


*
*
*  RO:
*    Modul software TempRent, proiect finanțat în cadrul POC, Axa prioritara 2 - Tehnologia Informației și Comunicațiilor (TIC) 
*    pentru o economie digitală competitivă, Prioritatea de investiții 2b - Dezvoltarea produselor și s
*    erviciilor TIC, a comerțului electronic și a cererii de TIC, cod SMIS 142474, 
*    Contractul de finanțare nr. 2/221_ap3/24.06.2021.
*


Created on Thu Jul 11 13:22:51 2019

@author: damian



TODO:
  - add EmbeddingEncoderModel as alternative to embedding-lookup and similarity model
  

"""

import tensorflow as tf
import os
import numpy as np
from collections import OrderedDict

from libraries import LummetryObject
from libraries import Logger
from time import time
from utils.utils import K_identity_loss, K_triplet_loss, simple_levenshtein_distance

__VER__ = '1.0.0'

class ALLANTaggerEngine(LummetryObject):
  """
  ALLAN 'Abstract' Engine
  """
  def __init__(self, log: Logger, 
               dict_word2index=None,
               dict_label2index=None,
               output_size=None,
               vocab_size=None,
               embed_size=None,
               model=None,
               DEBUG=False, MAX_CHR=100000,
               TOP_TAGS=None):
    if log is None or (type(log).__name__ != 'Logger'):
      raise ValueError("Loggger object is invalid: {}".format(log))
    #"".join([chr(0)] + [chr(i) for i in range(32, 127)] + [chr(i) for i in range(162,256)])
    self.MAX_CHR = MAX_CHR
    self.DEBUG = DEBUG
    self.min_seq_len = 20
    self.sess = None
    self.session = None
    self.trained = False
    self.pre_inputs = None
    self.pre_outputs = None
    self.pre_columns_end = None
    self.TOP_TAGS = TOP_TAGS
    self.prev_saved_model = []
    self.first_run = {}
    self.frames_data = None
    self.model_ouput = None
    self.embeddings = None
    self.generated_embeddings = None
    self.model = model
    self.embgen_model = None
    self.embgen_model_name = 'embgen_model'
    self.x_data_vocab = None
    self.output_size = len(dict_label2index) if dict_label2index is not None else output_size
    self.vocab_size = len(dict_word2index) if dict_word2index is not None else vocab_size
    self.dic_word2index = dict_word2index
    self.dic_labels = dict_label2index
    self.dic_topic2tags = None
    self.embed_size = embed_size
    self.emb_layer_name = 'emb_layer'
    super().__init__(log=log, DEBUG=DEBUG)
    return

  
  def startup(self):
    super().startup()
    self.__name__ = 'AT_TE'
    self.version = __VER__
    self.P("Init ALLANEngine v{}...".format(self.version))
    self.char_full_voc = "".join([chr(x) for x in range(self.MAX_CHR)])
    self.train_config = self.config_data.get('TRAINING', {})
    self.token_config = self.config_data.get('TOKENS', {})
    self.PAD_ID = self.token_config.get('PAD', 0)
    self.UNK_ID = self.token_config.get('UNK', 1)
    self.SOS_ID = self.token_config.get('SOS', 2)
    self.EOS_ID = self.token_config.get('EOS', 3)
    self.SPECIALS = [
        self.PAD_ID,
        self.UNK_ID,
        self.SOS_ID,
        self.EOS_ID,
        ]
    self.train_folder = self.train_config.get('FOLDER', '')
    self.embgen_model_config = self.config_data.get('EMB_GEN_MODEL', {})
    self.model_config = self.config_data.get('MODEL', {})
    self.doc_ext = self.train_config.get('DOCUMENT','')
    self.label_ext = self.train_config.get('LABEL', '')
    if self.TOP_TAGS is None:
      self.TOP_TAGS = self.config_data.get('TOP_TAGS',10)
    self.fn_word2idx = self.config_data.get('WORD2IDX','')
    self.fn_idx2word = self.config_data.get('IDX2WORD','')
    self.fn_labels2idx = self.config_data.get('LABEL2IDX')
    self.fn_topic2tags = self.config_data.get('TOPIC2TAGS')
    self.doc_size = self.model_config.get('DOC_SIZE','')
    self.model_name = self.model_config.get('NAME', '')
    self.dist_func_name = self.config_data.get('DIST_FUNC', 'l2')
    if self.dic_word2index is not None:
      self._get_reverse_word_dict()
      self._get_vocab_stats()    
    self._generate_idx2labels()
    
    if self.fn_topic2tags is not None:
      self.P("Loading topic2tags file '{}'".format(self.fn_topic2tags))
      if '.txt' in self.fn_topic2tags:
        self.dic_topic2tags = self.log.load_dict_from_data(self.fn_topic2tags)
      else:
        self.dic_topic2tags = self.log.load_pickle_from_data(self.fn_topic2tags)
      if self.dic_topic2tags is not None:
        sample = ["{}:{}".format(k,v) for i,(k,v) in enumerate(self.dic_topic2tags.items()) if i<3]
        self.P("Loaded topic2tags. Obs({}/{}): {}".format(
          len(sample),
          len(self.dic_topic2tags),
          sample
          ))
      else:
        self.P("WARNING: topic2tags NOT loaded.")

    return
  
  def set_labels_dict(self, labels):
    if isinstance(labels, dict):
      self.dic_labels = labels
      
    elif isinstance(labels, (list, tuple, np.ndarray)):
      unique_labels = set()
      for obs in labels:
        for label in obs:
          unique_labels.add(label)
      self.dic_labels = {k:v for v,k in enumerate(list(unique_labels))}
    else:
      raise ValueError("Unknown labels type {}".format(type(labels)))
                    
    self.output_size = len(self.dic_labels) 
    return
        
  
  def _setup_vocabs_and_dicts(self):
    self.P("Loading labels file '{}'".format(self.fn_labels2idx))
    if ".txt" in self.fn_labels2idx:
      dict_labels2idx = self.log.load_dict_from_data(self.fn_labels2idx)
    else:
      dict_labels2idx = self.log.load_pickle_from_data(self.fn_labels2idx)
    if dict_labels2idx is None:
      self.log.P("WARNING: No labels2idx dict found")
    
    dic_index2label = {v:k for k,v in dict_labels2idx.items()}
    self.dic_labels = dict_labels2idx
    self.dic_index2label = dic_index2label
    self._setup_vocabs(self.fn_word2idx, self.fn_idx2word)
    return
    
  
  def _setup_word_embeddings(self, embeds_filename=None):
    self.embeddings = None
    fn_emb = embeds_filename
    if fn_emb is None:
      fn_emb = self.model_config.get('EMBED_FILE', '')

    fn_emb = self.log.get_data_file(fn_emb)
      
    if os.path.isfile(fn_emb):
      self.P("Loading embeddings {}...".format(fn_emb[-25:]))
      self.embeddings = np.load(fn_emb, allow_pickle=True)
      self.P("Loaded embeddings: {}".format(self.embeddings.shape), color='y')
      self.emb_size = self.embeddings.shape[-1]
      self.vocab_size = self.embeddings.shape[-2]
    else:
      self.P("WARNING: Embed file '{}' does not exists. embeddings='None'".format(
          fn_emb))
      if self.emb_size == 0:
        raise ValueError("No `EMBED_SIZE` defined in config and embed file could not be loaded!")
    return  
  
  
  def _setup_similarity_embeddings(self, generated_embeds_filename=None):
    self.generated_embeddings = None
    fn_emb = generated_embeds_filename
    if fn_emb is None:
      fn_emb = self.embgen_model_config.get('EMBED_FILE', '')
    if self.log.get_data_file(fn_emb) is not None:
      self.P("Loading similarity embeddings {}...".format(fn_emb))
      self.generated_embeddings = self.log.load_pickle_from_data(fn_emb)
      self.P(" Loaded similarity embeddings: {}".format(self.generated_embeddings.shape))    
    else:
      self.P("WARNING: Embed file '{}' does not exists. generated_embeddings='None'".format(
          fn_emb))
      self._get_generated_embeddings()
    return  


  def _init_hyperparams(self, dict_model_config=None):
    if dict_model_config is not None:
      self.model_config = dict_model_config    
      self.P("Using external model parameters")

    self.seq_len = self.model_config.get('SEQ_LEN', None)
    if self.seq_len == 0:
      self.seq_len = None
    self.emb_size = self.model_config.get('EMBED_SIZE',0)
    self.emb_trainable = self.model_config('EMBED_TRAIN', True)
    self.model_columns = self.model_config['COLUMNS']

    if self.pre_inputs is not None:
      self.model_input = self.pre_inputs
    else:
      self.model_input = self.model_config['INPUT']


    if self.pre_outputs:
      self.model_output = self.pre_outputs
    else:
      self.model_output = self.model_config['OUTPUT']

    self.dropout_end = self.model_config.get('DROPOUT_CONCAT', 0.2 )
    self.end_fc = self.model_config['END_FC']    
    return
  
  
  def _get_generated_embeddings(self, x_data_vocab=None):
    if self.embgen_model is None:
      raise ValueError("`embgen_model` must be trained before generating embeddings")
    self.P("Inferring generated embeddings with embgen_model '{}'...".format(
      self.embgen_model_name), color='y')
    if x_data_vocab is None:
      if self.x_data_vocab is None:        
        x_data_vocab = self._convert_vocab_to_training_data()
      else:
        x_data_vocab = self.x_data_vocab
      
    np_embs = np.zeros((self.embeddings.shape), dtype=np.float32)
    lens = np.array([len(x) for x in self.x_data_vocab])
    unique_lens = np.unique(lens)
    t1 = time()
    iters = len(unique_lens)
    for i,unique_len in enumerate(unique_lens):
      print("\rInferring generated embeddings: {:.1f}%".format(
          ((i+1)/iters)*100), end='', flush=True)
      mask = lens == unique_len
      batch = self.x_data_vocab[mask].tolist()
      np_batch = np.array(batch)
      yhat = self.embgen_model.predict(np_batch)
      np_embs[mask]  = yhat
    print("\r",end='')
    self.generated_embeddings = np_embs
    if self.embeddings.shape != self.generated_embeddings.shape:
      raise ValueError("Embs {} differ from generated ones {}".format(
          self.embeddings.shape, self.generated_embeddings.shape))
    t2 = time()
    self.P("Done inferring generated embeddings in {:.1f}s.".format(t2-t1))
    fn = self.embgen_model_config['EMBED_FILE'] if 'EMBED_FILE' in self.embgen_model_config.keys() else "embgen_embeds.pkl"
    self.log.save_pickle_to_data(self.generated_embeddings, fn)
    return 
  
  
  def analize_vocab_and_data(self, compute_lens=False):
    self.P("Analyzing given vocabulary:")
    voc_lens = [len(self.dic_index2word[x]) for x in range(len(self.dic_index2word))]
    self.log.show_text_histogram(voc_lens, 
                               caption='Vocabulary word len distrib',
                               show_both_ends=True)
    if self.x_data_vocab is not None:
      data_lens = [len(x) for x in self.x_data_vocab]
      self.log.show_text_histogram(data_lens, 
                                 caption='Vocab-based {} obs'.format(len(data_lens)),
                                 show_both_ends=True)
      if compute_lens:
        self._vocab_lens = np.array(data_lens)
        self._unique_vocab_lens = np.unique(data_lens)
    else:
      self.P("x_data_vocab` is `none`")
    return voc_lens
    

  def _convert_vocab_to_training_data(self, min_word_size=5):
    if self.x_data_vocab is not None:
      self.P("WARNING: `x_data_vocab` already is loaded")
    self.P("Converting vocabulary to training data...")
    self.P(" Post-processing with min_word_size={}:".format(min_word_size))
    t1 = time()
    x_data = []
    for i_word in range(self.embeddings.shape[0]):
      if i_word in self.SPECIALS:
        x_data.append([i_word] + [self.PAD_ID]* min_word_size)
        continue
      else:
        x_data.append(self.word_to_char_tokens(self.dic_index2word[i_word], 
                                       pad_up_to=min_word_size))
    self.x_data_vocab = np.array(x_data)
    self.analize_vocab_and_data(compute_lens=True)
    self.P(" Training data unique lens: {}".format(self._unique_vocab_lens))
    t2 = time()
    self.P("Done generating vocab training data in {:.1f}s.".format(t2-t1))
    return self.x_data_vocab
  
  
  def get_vocab_training_data(self, min_word_size=5):
    self._convert_vocab_to_training_data(
                              min_word_size=min_word_size)
    return
    

  
  def _setup_vocabs(self, fn_words_dict=None, fn_idx_dict=None):
    if fn_words_dict is None:
      fn_words_dict = self.fn_word2idx
    if fn_idx_dict is None:
      fn_idx_dict = self.fn_idx2word
      
    self.P("Loading vocabs...")
    if ".txt" in fn_words_dict:
      dict_word2idx = self.log.load_dict_from_data(fn_words_dict)
    else:
      dict_word2idx = self.log.load_pickle_from_data(fn_words_dict)
    if dict_word2idx is None:
      self.P("No word2idx dict found. This wil be solved from reversing idx2word", color='y')
    else:
      self.P("Found word2idx[{}]".format(len(dict_word2idx)), color='g')

    if ".txt" in fn_idx_dict:
      dict_idx2word = self.log.load_dict_from_data(fn_idx_dict)
    else:
      dict_idx2word = self.log.load_pickle_from_data(fn_idx_dict)
      if type(dict_idx2word) in [list, tuple]:
        dict_idx2word = {i:v for i,v in enumerate(dict_idx2word)}
    if dict_idx2word is None:
      self.P("WARNING: No idx2word dict found")
    else:
      self.P("Loaded idx2word[{}]".format(len(dict_idx2word)), color='g')
      
    if (dict_word2idx is None) and (dict_idx2word is not None):
      dict_word2idx = {v:k for k,v in dict_idx2word.items()}
      
    if (dict_word2idx is not None) and (dict_idx2word is None):
      dict_idx2word = {v:k for k,v in dict_word2idx.items()}
      
    self.dic_word2index = dict_word2idx
    self.dic_index2word = dict_idx2word
    sample = ["{}:'{}'".format(i,self.dic_index2word[i]) for i in range(7)]
    self.P("Loaded dicts. I2W({}/{}): {}".format(
      len(sample),len(self.dic_index2word),
      ", ".join(sample)
      ))
    return
  
  
  
  def word_to_char_tokens(self, word, pad_up_to=0):
    _idxs = []
    for _ch in word:
      if _ch not in self.char_full_voc:
        raise ValueError("'{}' {} not in char_vocab[{}]".format(
            _ch, ord(_ch), len(self.char_full_voc)))
      else:
        _idxs.append(self.char_full_voc.index(_ch))
    #_idxs = [self.char_full_voc.index(_ch) for _ch in word]
    n_chr = len(_idxs)
    if n_chr < pad_up_to:
      nr_added = pad_up_to - n_chr
      _idxs += [0]* (nr_added)
    return _idxs
  
  def char_tokens_to_word(self, tokens):
    chars = [self.char_full_voc[x] for x in tokens if x != 0]
    return "".join(chars)
  
  def _setup_dist_func(self, func_name='cos'):
    if func_name == 'l2':
      func = lambda x,y: ((x-y)**2).sum(axis=-1)
    elif func_name == 'l1':
      func = lambda x,y: np.abs(x-y).sum(axis=-1)
    elif func_name == 'cos':
      func = lambda x,y: 1 - (x.dot(y) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y)))
    else:
      raise ValueError("Unknown distance function '{}'".format(func_name))
    return func
  
  def dist(self, target, source=None):
    if len(target.shape) > 1:
      raise ValueError("Target must be a emb vector. Received {}".format(
          target.shape))
    if source is None:
      source = self.embeddings
    f = self._setup_dist_func(self.dist_func_name)
    return f(source, target)
      
  
  def _get_approx_embed(self, word):
    return self.__get_approx_embed(word)
  
  def __get_approx_embed(self, word):
    """
    INITIAL APPROACH WAS NOT RELIABLE:
        1. get aprox embed via regression model
        2.1. calculate closest real embedding -> id 
          or
        2.2. send the embed directly to the mode
    
    CORRECT (CURRENT) APPROACH IS TO: 
      determine closest word based on second mebedding matrix (similarity word matrix)
        
    """
    char_tokens = np.array(self.word_to_char_tokens(word, pad_up_to=5)).reshape((1,-1))
    if self.run_in_cpu:
      with tf.device('/cpu:0'):
        res = self.embgen_model.predict(char_tokens)
    else:
      res = self.embgen_model.predict(char_tokens)
    return res.ravel()
  
  
  def _get_closest_idx(self, aprox_emb, top=1, np_embeds=None):
    """
     get closest embedding index
    """
    if  (self.embeddings is None) and (np_embeds is None):
      raise ValueError("Both emb matrices are none!")
    
    if np_embeds is None:
      np_embeds = self.embeddings
      
    dist = self.dist(target=aprox_emb, source=np_embeds)
    _mins = np.argsort(dist)
    if top == 1:
      _min = _mins[0]
    else:
      _min = _mins[:top]
    return _min
  
  
  def _get_closest_idx_and_distance(self, aprox_emb, top=1, np_embeds=None):
    """
     get closest embedding index
    """
    if  (self.embeddings is None) and (np_embeds is None):
      raise ValueError("Both emb matrices are none!")
    
    if np_embeds is None:
      np_embeds = self.embeddings

    dist = self.dist(target=aprox_emb, source=np_embeds)
    _mins = np.argsort(dist)
    _dist = dist[_mins]
    if top == 1:
      _min = _mins[0]
      _dst = _dist[0]
    else:
      _min = _mins[:top]
      _dst = _dist[:top]      
    return _min, _dst
  
  
  def _get_token_from_embed(self, np_embed):
    if self.embeddings is None:
      raise ValueError("Embeddings matrix is undefined!")
    matches = (self.embeddings == np_embed).sum(axis=-1) == len(np_embed)
    if np.any(matches):
      return np.argmax(matches)
    else:
      return -1
    
  def _get_tokens_from_embeddings(self, np_embeddings):
    tokens = []
    for np_embed in np_embeddings:
      tokens.append(self._get_token_from_embed(np_embed))
    return tokens
  
  
  def get_unk_word_similar_id(self, unk_word, top=1, raw_test_mode=False):
    if unk_word in self.dic_word2index.keys():
      self.P("WARNING: presumed '{}' unk word is already in vocab!".format(unk_word))
    if self.generated_embeddings is None:
      raise ValueError("`generated_embeddings` matrix must be initialized before calculating unk word similarity")
    # first compute generated embedding
    aprox_emb = self.__get_approx_embed(unk_word)
    if top > 1 or raw_test_mode:
      # get closest words id from the generated embeddings the ids will be the same in the real embeddings matrix
      idx = self._get_closest_idx(aprox_emb=aprox_emb, top=top, np_embeds=self.generated_embeddings)
    else:      
      # now we get top 10 and return most similar
      idxs = self._get_closest_idx(aprox_emb=aprox_emb, top=10, np_embeds=self.generated_embeddings)
      words = [self.dic_index2word[x] for x in idxs]
      min_dist = 1000
      candidate = words[0]
      for word in words:
        # calc levenshtein distance
        dist = simple_levenshtein_distance(unk_word, word)
        if dist <= min_dist:
          candidate = word
          min_dist = dist
      idx = self.dic_word2index[candidate]      
    return idx
  
  
  def get_unk_word_similar_word(self, unk_word, top=1, raw_test_mode=False, debug=False):
    ids = self.get_unk_word_similar_id(unk_word, top=top, raw_test_mode=raw_test_mode)
    if type(ids) is np.ndarray:
      _result = [self.dic_index2word[x] for x in ids]
    else:
      _result = self.dic_index2word[ids]
    if debug:
      if isinstance(_result,str):
        _result = [_result]
      _result = [(x, simple_levenshtein_distance(unk_word, x)) for x in _result]
    return _result
  
  
  def get_similar_words_by_text(self, word, top=1):
    idx = self.dic_word2index[word]
    embed = self.embeddings[idx]
    idxs = self._get_closest_idx(aprox_emb=embed, top=top)
    if type(idxs) is np.ndarray:
      _result = [self.dic_index2word[x] for x in idxs]
    else:
      _result = self.dic_index2word[idxs]
    return _result
  

  def get_similar_words_by_id(self, id_word, top=1):
    embed = self.embeddings[id_word]
    idxs = self._get_closest_idx(aprox_emb=embed, top=top)
    if type(idxs) is np.ndarray:
      _result = [self.dic_index2word[x] for x in idxs]
    else:
      _result = self.dic_index2word[idxs]
    return _result

  
  def _word_encoder(self, word, convert_unknown_words=False, raw_conversion=False):
    if self.embeddings is None:
      self._setup_word_embeddings()
      if self.embeddings is None:
        raise ValueError("Embeddings loading failed!")
    idx = self.dic_word2index.get(word, self.UNK_ID)
    if convert_unknown_words and (idx == self.UNK_ID):
      idx = self.get_unk_word_similar_id(word, raw_test_mode=raw_conversion)
    if idx in self.SPECIALS:
      idx = self.UNK_ID      
    emb = self.embeddings[idx]
    return idx, emb
      
  
  def _get_reverse_word_dict(self):
    self.P("Constructing reverse vocab...")
    self.dic_index2word = {v:k for k,v in self.dic_word2index.items()}
  
  def _get_words(self, text):
    lst_splitted = tf.keras.preprocessing.text.text_to_word_sequence(text)
    return lst_splitted
  

  def encode(self, text, 
             text_label=None, 
             to_onehot=True,
             rank_labels=False,
             convert_unknown_words=True,
             direct_embeddings=False,
             fixed_len=0,
             raw_conversion=False,
             DEBUG=False):
    """
    this function will tokenize or directly output the embedding represenation
    of the input list of documents together with the given labels for each
    document (if labels are provided)
    """
    s = "Starting text corpus conversion"
    if direct_embeddings:
      s += ' into embeddings'
    else:
      s += ' into tokens'
      
    if convert_unknown_words:
      s += ' and converting unknown words'
      if direct_embeddings:
        s += ' into embeddings'
      else:
        s += ' into similar tokens'
    if DEBUG:
      self.P(s)
    if fixed_len and DEBUG:
      self.P("Sequences less then {} will pe padded and those above will be truncated".format(fixed_len))
    if type(text) in [str]:
      text = [text]
    lst_enc_texts = []
    lst_enc_labels = []
    self.last_max_size = 0
    nr_obs = len(text)
    for i,txt in enumerate(text):
      if not DEBUG and (len(text) > 10):
        print("\rProcessing {:.1f}% of input documents...".format(
            i/nr_obs * 100), end='', flush=True)
      if isinstance(txt, list):
        # text is already splitted. thank you :)
        splitted = txt
      else:
        splitted = self._get_words(txt)
      self.last_max_size = max(self.last_max_size, len(splitted))
      tkns = []
      embs = []
      for word in splitted:
        tk,em = self._word_encoder(word, 
                                   convert_unknown_words=convert_unknown_words,
                                   raw_conversion=raw_conversion,
                                   )
        tkns.append(tk)
        embs.append(em)

      if direct_embeddings:
        tokens = embs
      else:
        tokens = tkns
      if DEBUG:
        self.P("Converted:")
        self.P("  '{}'".format(text))
        self.P(" into")
        self.P("  '{}'".format(self.decode(tkns))) 
      if len(tokens) < fixed_len:
        added = fixed_len - len(tokens)
        if direct_embeddings:
          tokens += [self.embeddings[self.PAD_ID]] * added
        else:
          tokens += [self.PAD_ID] * added
      if fixed_len > 0:
        tokens = tokens[:fixed_len]
      lst_enc_texts.append(tokens)
    if text_label is not None:
      assert type(text_label) in [list, tuple, np.ndarray], "labels must be provided as list/list or lists"
      if self.dic_labels is None:
        self.P("WARNING: not labels dictionary - using current set of {} observations as label basis".format(
          len(text_label)))
        self.set_labels_dict(text_label)
      if type(text_label[0]) in [str]:
        text_label = [text_label]
      if to_onehot:
        lst_enc_labels = self.labels_to_onehot_targets(text_label, 
                                                       rank=rank_labels,
                                                       DEBUG=DEBUG)
      else:
        for lbl in text_label:
          l_labels =[self.dic_labels[x] for x in lbl]
          lst_enc_labels.append(l_labels)
    if direct_embeddings:
      lst_enc_texts = np.array(lst_enc_texts)
    if len(lst_enc_labels) > 0:
      return lst_enc_texts, lst_enc_labels
    else:
      return lst_enc_texts

    
  def decode(self, tokens, 
             tokens_as_embeddings=False,
             labels_idxs=None, 
             labels_from_onehot=True):
    """
    this function will transform a series of token sequences into text as well 
    as a list of sequences of labels indices into coresponding indices
    """
    if (("int" in str(type(tokens[0]))) or 
        (type(tokens) == np.ndarray and len(tokens.shape) == 2)):
      tokens = [tokens]
    texts = []
    labels = []
    for seq in tokens:
      if tokens_as_embeddings:
        seq = self._get_tokens_from_embeddings(seq)
      txt = " ".join([self.dic_index2word[x] for x in seq if x != self.PAD_ID]) 
      texts.append(txt)
    if labels_idxs is not None:
      if type(labels_idxs[0]) in [int]:
        labels_idxs = [labels_idxs]
      for seq_idxs in labels_idxs:
        if labels_from_onehot:
          seq = np.argwhere(seq_idxs).ravel().tolist()
        else:
          seq = seq_idxs
        c_labels = [self.dic_index2labels[x] for x in seq]
        labels.append(c_labels)
      return texts, labels
    return texts

  @property
  def loaded_vocab_size(self):
    return len(self.dic_word2index)  
  
  @property
  def loaded_labels_size(self):
    return len(self.dic_labels)
  
  def one_hotter(self, data):
    return tf.keras.utils.to_categorical(data, num_classes=self.output_size)
  
  def labels_to_onehot_targets(self, labels, rank=False, DEBUG=False):
    if not type(labels[0]) in [list, tuple, np.ndarray]:
      raise ValueError("labels must be provided as list of lists or 2d ndarray")
    idx_labels = []
    if type(labels[0][0]) is str:
      idx_labels = [[self.dic_labels[x] for x in obs] for obs in labels]
      need_one_hot = True
    else:
      idx_labels = labels
      need_one_hot = False
    maxes = [max(x) for x in idx_labels]
    sizes = [len(x) for x in idx_labels]
    
    if need_one_hot:
      if DEBUG:
        self.P("Converting labels to targets")
      lst_outs = []
      for obs in idx_labels:
        np_obs = np.array([self.one_hotter(x) for x in obs])
        np_obs = np_obs.sum(axis=0)
        if rank:
          np_obs[obs] = np.linspace(1, 0.6, num=len(obs))
        lst_outs.append(np_obs)
      np_output = np.array(lst_outs)
    elif np.unique(sizes).size != 1:
      raise ValueError("something is wrong, labels are one-hot but vector sizes differ!")
    else:
      np_output = np.array(idx_labels)
      self.P("Everything looks good, no processing required on {}".format(
          np_output.shape))
    return np_output
  
  def _predict_single(self, tokens_or_embeddings):
    """
    given a vector of tokens of embeddings (matrix) will infer the
    tags
    """
    shape = tokens_or_embeddings.shape
    #dtype = tokens_or_embeddings.dtype
    if len(shape) == 1:
      tokens_or_embeddings = tokens_or_embeddings.reshape((1,-1))
    if len(shape) == 2:
      # predict on integer tokens
      pass
    if (len(shape) == 3):
      # predict on embeds
      req_emb_size = shape[-1]
      model_inp = self.model.inputs[0].shape[-1]
      if  model_inp != req_emb_size:
        raise ValueError("Attempted to feed direct embeds obs {} in model inputs {}".format(
            shape, self.model.inputs[0]))
    x_input = tokens_or_embeddings
    if x_input.shape[1] < self.min_seq_len:
      raise ValueError("Cannot call model.predict on seq less than {} tokens".format(
          self.min_seq_len))
    preds = self.model.predict(x_input)
    return preds

  
  def predict_text(self, 
                   text, 
                   convert_unknown_words=True,
                   convert_tags=True,
                   top=None,
                   return_input_processed=True,
                   return_topic=True,
                   force_below_threshold=True,
                   DEBUG=False,
                   verbose=1):
    """
    given a simple document will output the results based on curent model
      Args:
        text : the document that can be one string or a list of strings
        convert_unknown_words : True will use siamse net to find unk words
        convert_tags : True will convert tag-id into tag names
        top : number of top findings (5)
      
      Returns:
        the found tags dict in {tag: proba ...} format
    """
    if top is None:
      top = self.TOP_TAGS
    assert self.trained and self.model is not None
    self.maybe_generate_idx2labels()
    if DEBUG: 
      self.P("Inferring initial text '{}'".format(text))
    direct_embeddings = False
    if len(self.model.inputs[0].shape) == 3:
      direct_embeddings = True
      if DEBUG:
        self.P("Model inputs {} identified to directly receive embeddings".format(
            self.model.inputs[0].shape))
    
    tokens = self.encode(text, 
                         convert_unknown_words=convert_unknown_words,
                         direct_embeddings=direct_embeddings,
                         fixed_len=self.doc_size,
                         DEBUG=DEBUG)
    processed_input = self.decode(tokens=tokens, tokens_as_embeddings=direct_embeddings)[0]
    if verbose >= 1:
      self.P("Inferring (decoded): '{}'".format(processed_input))
    np_tokens = np.array(tokens)
    np_tags_probas = self._predict_single(np_tokens)
    tags = self.array_to_tags(np_tags_probas, 
                              top=top, 
                              convert_tags=convert_tags,
                              force_below_threshold=force_below_threshold)
    
    topic_document = None
    if return_topic:
      topic_document = self.find_topic(dict_tags=tags,
                                       choose_by_length=False) #USE True to check by length    
    
    
    if DEBUG:
      top_10_preds = self.array_to_tags(
                                        np_tags_probas, 
                                        top=10, 
                                        convert_tags=True,
                                        force_below_threshold=True)
      self.P("  Predicted: {}".format("".join(["'{}':{:.3f} ".format(k,v) 
                                    for k,v in top_10_preds.items()])))
      
    ret = (tags,)
    if return_topic:
      ret += (topic_document,)
    if return_input_processed:
      ret += (text, processed_input)
    
    return ret
  
  
  def array_to_tags(self, np_probas, top=5, convert_tags=True, force_below_threshold=False):
    threshold = 0.5 if "tag" in self.model_output else 0
    np_probas = np_probas.ravel()
    tags_idxs = np.argsort(np_probas)[::-1]
    top_idxs = tags_idxs[:top]
    top_labels = [self.dic_index2label[idx] for idx in top_idxs]
    top_prob = np_probas[top_idxs]
    self.last_probas = top_prob
    self.last_labels = top_labels
    dct_res = OrderedDict()
    for i, idx in enumerate(top_idxs):
      if not force_below_threshold:
        if (i > 0) and (np_probas[idx] < threshold):
          # skip after first if below threshold
          continue
      if convert_tags:
        dct_res[self.dic_index2label[idx]] = float(np_probas[idx])
      else:
        dct_res[idx] = float(np_probas[idx])
    return dct_res
  
  def maybe_generate_idx2labels(self):
    if self.dic_index2label is None:
      self._generate_idx2labels()
    return
         
  def _generate_idx2labels(self):
    if self.dic_labels is not None:
      self.dic_index2label = {v:k for k,v in self.dic_labels.items()}
    else:
      self.dic_index2label = None
    return
  
  def get_stats(self, X_tokens, show=True, X_docs=None, X_labels=None):
    self.P("Calculating documens stats...")
    sizes = [len(seq) for seq in X_tokens]
    idxs_min = np.argsort(sizes)
    dict_stats = {
        "Min" : int(np.min(sizes)), 
        "Max" : int(np.max(sizes)), 
        "Avg" : int(np.mean(sizes)),
        "Med" : int(np.median(sizes)),
        }
    self.P("Done calculating documents stats.")
    if show:
      for stat in dict_stats:
        self.P("  {} docs size: {}".format(stat, dict_stats[stat]))
      self.P("  Example of small docs:")
      for i in range(5):
        idx = idxs_min[i]
        i_sz = sizes[idx]
        if X_docs is not None:
          s_doc = X_docs[idx]
        else:
          s_doc = self.decode(X_tokens[idx], 
                              tokens_as_embeddings=self.direct_embeddings)
        lbl = ''
        if X_labels is not None:
          lbl = 'Label: {}'.format(X_labels[idx])
        self.P("    ID:{:>4} Size:{:>2}  Doc: '{}'  {}".format(
            idx, i_sz, s_doc, lbl))
    return dict_stats
  
  def __pad_data(self, X_tokens, max_doc_size=None):
    """
     pad data based on 'max_doc_size' or on predefined self.max_doc_size
    """
    if max_doc_size is None:
      max_doc_size = self.max_doc_size
    self.P("Padding data...")
    self.get_stats(X_tokens)
    X_data = tf.keras.preprocessing.sequence.pad_sequences(
        X_tokens, 
        value=self.PAD_ID,
        maxlen=max_doc_size, 
        padding='post', 
        truncating='post')
    self.P("Data padded to {}".format(X_data.shape))
    return X_data
    

  def _train_loop(self, X_data, y_data, batch_size, n_epochs, 
                  X_text_valid=None, y_text_valid=None,
                  save_best=True,
                  save_end=True, 
                  test_every_epochs=1,
                  DEBUG=True,
                  compute_topic=True):
    """
    this is the basic 'protected' training loop loop that uses tf.keras methods and
    works both on embeddings inputs or tokenized inputs
    """
    n_obs = len(X_data)
    self.P("Training on {} obs, {} epochs, batch {}".format(
        n_obs,n_epochs, batch_size))
    n_batches = n_obs // batch_size + 1
    self.train_losses = []
    self.log.SupressTFWarn()
    best_recall = 0
    self.train_recall_history = []
    self.train_recall_history_epochs = []
    self.train_recall_non_zero_epochs = []
    self.train_epoch = 0
    
    fct_test = self.test_model_on_texts_with_topic if compute_topic else self.test_model_on_texts
    
    for epoch in range(n_epochs):
      self.train_epoch = epoch + 1
      epoch_losses = []
      for i_batch in range(n_batches):
        batch_start = (i_batch * batch_size) % n_obs
        batch_end = min(batch_start + batch_size, n_obs)
        X_batch = np.array(X_data[batch_start:batch_end].tolist())
        y_batch = np.array(y_data[batch_start:batch_end])
        batch_output = self.model.train_on_batch(X_batch, y_batch)
        s_bout = self.log.EvaluateSummary(self.model, batch_output)
        loss = batch_output[0] if type(batch_output)  in [list, np.ndarray, tuple] else batch_output
        batch_info = "Epoch {:>3}: {:>5.1f}% completed [{}]".format(
            epoch+1, i_batch / n_batches * 100, s_bout)        
        print("\r {}".format(batch_info), end='', flush=True)
        self.train_losses.append(loss)
        epoch_losses.append(loss)        
        self.trained = True
      print("\r",end="")
      epoch_loss = np.mean(epoch_losses)
      self.P("Epoch {} done. loss:{:>7.4f}, all avg :{:>7.4f}, last batch: [{}]".format(
          epoch+1, epoch_loss,np.mean(self.train_losses), s_bout))
      if (epoch > 0) and (test_every_epochs > 0) and (X_text_valid is not None) and ((epoch+1) % test_every_epochs == 0):
        self.P("Testing on epoch {}".format(epoch+1))
        rec = fct_test(lst_docs=X_text_valid, lst_labels=y_text_valid,
                       DEBUG=True, top=10)
        if compute_topic:
          rec, topic_rec = rec
        if self.last_test_non_zero and (best_recall < rec):
          self.train_recall_non_zero_epochs.append(epoch+1)
          s_name = 'ep{}_R{:.0f}_ANZ'.format(epoch+1, rec)
          self.save_model(s_name, delete_prev_named=True)
          best_recall = rec
        elif best_recall < rec:
          s_name = 'ep{}_R{:.0f}'.format(epoch+1, rec)
          self.save_model(s_name, delete_prev_named=True)
          best_recall = rec
     
          
    self.P("Model training done.")
    self.P("Train recall history: {}".format(self.train_recall_history))
    if compute_topic:
      self.P("Train topic recall history: {}".format(self.train_topic_recall_history))
    self._reload_embeds_from_model()
    if save_end:
      self.save_model()    
    return  
  
  
  def save_model(self, name=None, delete_prev_named=False, DEBUG=False):
    s_name = self.model_name
    if name is not None:
      s_name += '_' + name
      
    debug = (not delete_prev_named) or DEBUG
    
    if debug:      
      self.P("Saving tagger model '{}'".format(s_name))
    fn = os.path.join(self.log.get_models_folder(), l.file_prefix + '_' + s_name +'.h5')
    self.model.save(fn)

    if delete_prev_named:
      if self.prev_saved_model != []:
        new_list = []
        for _f in self.prev_saved_model:
          if os.path.isfile(_f):
            try:
              os.remove(_f)              
            except:
              new_list.append(_f)
        self.prev_saved_model = new_list
      self.prev_saved_model.append(fn)
    return
  
  
  def _check_model_inputs(self):
    if len(self.model.inputs[0].shape) == 3:
      self.direct_embeddings = True
      self.P("Model inputs {} identified to directly receive embeddings".format(
          self.model.inputs[0].shape))
    else:
      self.direct_embeddings = False
      self.P("Model inputs {} identified to receive tokens".format(
          self.model.inputs[0].shape))
    return
      
  
  def train_on_texts(self, 
            X_texts, 
            y_labels, 
            X_texts_valid=None,
            y_labels_valid=None,
            convert_unknown_words=True,
            batch_size=32, 
            n_epochs=1,
            save=True,
            skip_if_pretrained=True, 
            test_every_epochs=5,
            DEBUG=True,   
            compute_topic=True
            ):
    """
    this methods trains the loaded/created `model` directly on text documents
    and text labels after tokenizing and (if required) converting to embeddings 
    the inputs all based on the structure of the existing `model` inputs
    """
    
    if self.model is None:
      raise ValueError("Model is undefined!")
    
    if self.model_output is None:
      raise ValueError("Model output config must be defined")
    
    if skip_if_pretrained and self.trained:
      self.P("Skipping training...")
      return
    if not (type(X_texts) in [list, tuple]):
      raise ValueError("Train function expects X_texts as a list of text documents")
      
    if not (type(y_labels) in [list, tuple]) or (type(y_labels[0]) not in [list, tuple]):
      raise ValueError("Train function expects y_labels as a list of label lists")
    n_obs = len(X_texts)
    if n_obs != len(y_labels):
      raise ValueError("X and y contain different number of observations")

    self._check_model_inputs()
    
    if convert_unknown_words and self.generated_embeddings is None:
      self.setup_embgen_model()
      

    rank_labels = 'multi' in self.model_output
    
    
    pad_data = self.doc_size
    
    X_tokens, y_data = self.encode(X_texts, 
                                   text_label=y_labels,
                                   to_onehot=True,
                                   rank_labels=rank_labels,
                                   convert_unknown_words=convert_unknown_words,
                                   direct_embeddings=self.direct_embeddings,
                                   fixed_len=pad_data)

    self.max_doc_size = self.doc_size

    self.P("Training on sequences of max {} words".format(self.max_doc_size))

    if pad_data > 0:
      X_data = X_tokens
    else:
      batch_size = 1
      X_data = X_tokens
      self.get_stats(X_data, X_labels=y_labels)
      self.P("Reducing batch_size to 1 and processing doc by doc")

    idxs_chk = [133] + np.random.choice(n_obs, size=5, replace=False).tolist()
    if X_texts_valid is None:
      X_texts_valid = [X_texts[x] for x in idxs_chk]
      y_labels_valid = [y_labels[x] for x in idxs_chk]
      
    if self.direct_embeddings:
      self.P("Sanity check before training loop for direct embeddings:")
      for idx in idxs_chk:
        x_e = X_data[idx]
        y_l = y_data[idx]
        txt = X_texts[idx]
        lbl = y_labels[idx]
        x_txt = self.decode(tokens=x_e, tokens_as_embeddings=True)
        y_lbl = self.array_to_tags(y_l)
        self.P("  Doc: '{}'".format(txt))
        self.P("  DEC: '{}'".format(x_txt[0]))
        self.P("  Lbl:  {}".format(lbl))
        self.P("  DEC:  {}".format(y_lbl))
        self.P("")

    
    self._train_loop(X_data, y_data, batch_size, n_epochs, 
                     X_text_valid=X_texts_valid, y_text_valid=y_labels_valid,
                     save_best=save, save_end=save, test_every_epochs=test_every_epochs,
                     compute_topic=compute_topic)
    
    if compute_topic:
      return self.train_recall_history, self.train_topic_recall_history
    return self.train_recall_history



  def __train_on_tokens(self, 
                      X_tokens, 
                      y_labels,
                      batch_size=32, 
                      n_epochs=1,
                      save=True,
                      skip_if_pretrained=False):
    """
    TODO: Check this one and make it public!
    
    this method assumes a `model` has been created and it accepts
    sequences of tokens as inputs. y_labels are indices of labels
    """
    if self.model_ouput is None:
      raise ValueError("Model output config must be defined")
      
    if skip_if_pretrained and self.trained:
      self.P("Skipping training...")
    if not (type(X_tokens) in [list, tuple, np.ndarray]):
      raise ValueError("Train function expects X_texts as a ndarray or list-of-lists of tokens")
      
    if type(y_labels) not in [list, tuple, np.ndarray]:
      raise ValueError("Train function expects y_labels as a ndarray or list of label lists")
      
    n_obs = len(X_tokens)
    if n_obs != len(y_labels):
      raise ValueError("X and y contain different number of observations")
    
    rank_labels = 'multi' in self.model_output
    
    y_data = self.labels_to_onehot_targets(y_labels, rank=rank_labels)

    self.max_doc_size = self.doc_size
    
    self.P("Training on sequences of max {} words".format(self.max_doc_size))

    # TODO: must implement embedding generation for proposed tokenized  data
    self._check_model_inputs()
    ###
    
    if force_batch:
      X_data = self.pad_data(X_tokens=X_tokens)
    else:
      batch_size = 1
      X_data = X_tokens
      self.P("Reducing batch_size to 1 and processing doc by doc")

    self._train_loop(X_data, y_data, batch_size, n_epochs, 
                     save_best=save, save_end=save)
    return
  
  
  def _reload_embeds_from_model(self,):
    self.P("Reloading embeddings from model")
    self._check_model_inputs()
    if self.direct_embeddings:
      self.P("Skip reload: Cannot reload embeddings (input is: {}, second layer: {}".format(
          self.model.inputs[0].shape, self.model.layers[1].__class__.__name__))
      return
    lyr_emb = None
    for lyr in self.model.layers:
      if lyr.name == self.emb_layer_name:
        lyr_emb = lyr
    if lyr_emb is None:
      raise ValueError("Embeddings layer not found!")
    self.embeddings = lyr_emb.get_weights()[0]
    self.P("Embeddings reloaded from model {}".format(
        self.embeddings.shape))
    
  def maybe_load_pretrained(self):
    _res = False
    if "PRETRAINED" in self.model_config.keys():
      fn = self.model_config['PRETRAINED']
      _ver = ''
      _f = 0
      for i,x in enumerate(fn):
        if x.isdigit():
          _ver += x
        if x == '_' and i != 0 :
          if _f == 0:
            _ver += "."
            _f += 1
          else:
            break
      self.version += '.' + _ver
      if self.log.GetModelsFile(fn) is not None:
        self.P("Loading pretrained model {}".format(fn))
        self.model = self.log.LoadKerasModel(
                                  fn,)
        _res = True
        self._reload_embeds_from_model()
    return _res
  
  
  def maybe_load_pretrained_embgen(self, embgen_model_file=None, run_in_cpu=False):
    _res = False
    self.run_in_cpu = run_in_cpu
    if "PRETRAINED" in self.embgen_model_config.keys() or embgen_model_file is not None:
      fn = self.embgen_model_config['PRETRAINED'] if embgen_model_file is None else embgen_model_file
      fn_model = self.log.get_models_file(fn)
      if fn_model is not None:
        self.P("Loading pretrained embgen model {}".format(fn), color='y')
        if self.run_in_cpu:
          with tf.device('/cpu:0'):
            self.embgen_model = tf.keras.models.load_model(
              filepath=fn_model,
              custom_objects=None,
              )
        else:
          self.embgen_model = tf.keras.models.load_model(
            filepath=fn_model,
            custom_objects=None,
            )          
        self.embgen_model_name = fn
        _res = True
      else:
        self.P("WARNING: attempt to load pretrained embgen '{}' failed.".format(fn))
    return _res
  
  
  def setup_embgen_model(self, embgen_model_file=None, generated_embeds_filename=None, run_in_cpu=False):
    self.maybe_load_pretrained_embgen(embgen_model_file=embgen_model_file, run_in_cpu=run_in_cpu)
    self._setup_similarity_embeddings(generated_embeds_filename=generated_embeds_filename)
    return
  

  def setup_pretrained_model(self):
    if self.maybe_load_pretrained():
      self.P("Pretrained model:\n{}".format(
          self.log.get_keras_model_summary(self.model)))
      self.trained = True
    return  
    

  def _get_vocab_stats(self,):
    if self.dic_word2index is None:
      raise ValueError("Vocab dict is not available !")
    if self.dic_index2word is None:
      raise ValueError("Reverse vocab dict is not available !")
    lens = [len(k) for k in self.dic_word2index]
    dct_stats = {
          "Max" : int(np.max(lens)),
          "Avg" : int(np.mean(lens)),
          "Med" : int(np.median(lens)),
        }
    self.P("Loaded vocab:")
    for k in dct_stats:
      self.P("  {} word size: {}".format(k, dct_stats[k]))
    self.log.ShowTextHistogram(lens)
    return dct_stats
  
  def check_labels_set(self, val_labels):
    for obs in val_labels:
      if type(obs) not in [list, tuple, np.ndarray]:
        raise ValueError("LabelSetCheck: All observations must be lists of labels")
      for label in obs:
        if label not in self.dic_labels.keys():
          raise ValueError("LabelSetCheck: Label '{}' not found in valid labels dict".format(label))
    self.P("LabelSetCheck: All {} labels are valid.".format(len(val_labels)))
    return
    
  def initialize(self):
    self.P("Full initialization started ...")
    self._setup_vocabs_and_dicts()
    self._init_hyperparams()
    self._setup_word_embeddings()
    self.setup_embgen_model()
    self.setup_pretrained_model()    
    if self.embeddings is None:
      raise ValueError("Embeddings loading failed!")
    if self.model is None:
      raise ValueError("Model loading failed!")
    if self.embgen_model is None:
      raise ValueError("EmbGen model loading failed!")
    if self.generated_embeddings is None:
      raise ValueError("Generated similarity siamese embeddings loading failed!")
    self.P("Full initialization done {} v{}.".format(
        self.__class__.__name__, self.version))
    
    
  def tagdict_to_text(self, tags, max_tags=None):
    txt = ''
    cnt = 0
    for k in tags:
      cnt += 1
      txt = txt + "'{}':{:.2f} ".format(k, tags[k])
      if max_tags is not None:
        if cnt >= max_tags:
          break
    return txt

  
  def test_model_on_texts(self, lst_docs, lst_labels, top=5, 
                          show=True, DEBUG=False, record_trace=True, zero_penalty=-1.0):
    """
    function that calculates (and displays) model validation/testing indicators
    
    inputs:
      lst_docs    : list of documents (each can be a string or a list of strings)
      lst_labels  : list of labels (list) for each document
      top         : max number of tags to generate 
      show        : display stats
    
    returns:
      scalar float with overall accuracy (mean recall)
      
    """
    if not hasattr(self, "train_recall_history"):
      self.train_recall_history = []
    if type(lst_docs) == str:
      lst_docs = [lst_docs]
    if type(lst_labels[0]) == str:
      lst_labels = [lst_labels]
    if len(lst_docs) != len(lst_labels):
      raise ValueError("Number of documents {} must match number of label-groups {}".format(
          len(lst_docs), len(lst_labels)))
    docs_acc = []
    tags_per_doc = []
    if show:
      self.P("")
      self.P("Starting model testing on {} documents with zero-doc-penalty: {:.1f}".format(
          len(lst_docs), zero_penalty))
    zero_preds = False
    self.last_test_non_zero = False
    for idx, doc in enumerate(lst_docs):
      doc_acc = 0
      dct_tags, inputs = self.predict_text(doc, convert_tags=True, 
                                   convert_unknown_words=True, 
                                   top=top,
                                   DEBUG=False,
                                   return_input_processed=True,
                                   verbose=0,
                                   )
      lst_tags = [x.lower() for x in dct_tags]
      gt_tags = lst_labels[idx]
      for y_true in gt_tags:
        if y_true.lower() in lst_tags:
          doc_acc += 1
      if show and DEBUG:
        self.P("  Inputs: ({} chars) '{}...'".format(len(inputs), inputs[:50]))
        self.P("  Predicted: {}".format(self.tagdict_to_text(dct_tags, max_tags=5)))
        self.P("  Labels:    {}".format(gt_tags[:5]))
        self.P("  Match: {}/{}".format(doc_acc, len(gt_tags)))
        self.P("")
      doc_prc = doc_acc / len(gt_tags)      
      tags_per_doc.append(len(gt_tags))
      if doc_prc == 0:
        zero_preds = True
        doc_prc = zero_penalty
      docs_acc.append(doc_prc)
    overall_acc = np.mean(docs_acc)
    self.last_test_non_zero = not zero_preds
    if record_trace:
      self.train_recall_history.append(round(overall_acc * 100, 1))
      self.train_recall_history_epochs.append(self.train_epoch)
    if show:
      self.P("Tagger benchmark on {} documents with {:.1f} avg tags/doc".format(
          len(lst_docs), np.mean(tags_per_doc)))
      self.P("  {}".format("ZERO PREDS :( !!!" if zero_preds else "Hurray! All preds non-zero! :)))"))
      self.P("  Overall recall: {:5.1f}%".format(overall_acc * 100))
      self.P("  Max doc recall: {:5.1f}%".format(np.max(docs_acc) * 100))
      self.P("  Min doc recall: {:5.1f}%".format(np.min(docs_acc) * 100))
      self.P("  Med doc recall: {:5.1f}%".format(np.median(docs_acc) * 100))
    return max(0, round(overall_acc * 100, 2))
  
  
  def test_model_on_texts_with_topic(self, lst_docs, lst_labels, top=10, 
                                     show=True, DEBUG=False, record_trace=True, zero_penalty=-1.0):
    """
    function that calculates (and displays) model validation/testing indicators
    
    inputs:
      lst_docs    : list of documents (each can be a string or a list of strings)
      lst_labels  : list of labels (list) for each document
      top         : max number of tags to generate 
      show        : display stats
    
    returns:
      scalar float with overall accuracy (mean recall)
      
    """
    if not hasattr(self, "train_recall_history"):
      self.train_recall_history = []
    if not hasattr(self, "train_topic_recall_history"):
      self.train_topic_recall_history = []
    if type(lst_docs) == str:
      lst_docs = [lst_docs]
    if type(lst_labels[0]) == str:
      lst_labels = [lst_labels]
    if len(lst_docs) != len(lst_labels):
      raise ValueError("Number of documents {} must match number of label-groups {}".format(
          len(lst_docs), len(lst_labels)))
    if self.dic_topic2tags is None:
      raise ValueError("Each topic should have a list of associated tags.")
    docs_acc = []
    tags_per_doc = []
    if show:
      self.P("")
      self.P("Starting model testing on {} documents with zero-doc-penalty: {:.1f}".format(
          len(lst_docs), zero_penalty))
    zero_preds = False
    self.last_test_non_zero = False
    
    matched_by_topic = 0
    
    for idx, doc in enumerate(lst_docs):
      doc_acc = 0
      dct_tags, inputs = self.predict_text(doc, convert_tags=True, 
                                   convert_unknown_words=True, 
                                   top=top,
                                   DEBUG=False,
                                   return_input_processed=True,
                                   verbose=0,
                                   )
      
      pred_topic = self.find_topic(dct_tags, choose_by_length=False)
      
      lst_tags = [x.lower() for x in dct_tags]
      gt_tags = lst_labels[idx]
      true_topic = list(filter(lambda x: 'topic' in x, gt_tags))
      assert len(true_topic) == 1
      true_topic = true_topic[0]
      
      if pred_topic == true_topic:
        matched_by_topic += 1
      
      for y_true in gt_tags:
        if y_true.lower() in lst_tags:
          doc_acc += 1
      if show and DEBUG:
        self.P("  Inputs: ({} chars) '{}...'".format(len(inputs), inputs[:50]))
        self.P("  Predicted: {}".format(self.tagdict_to_text(dct_tags, max_tags=10)))
        self.P("  Labels:    {}".format(gt_tags[:5]))
        self.P("  Match: {}/{}".format(doc_acc, len(gt_tags)))
        self.P("  True Topic: {}".format(true_topic))
        self.P("  Pred Topic: {}".format(pred_topic))
        self.P("  Correct Topic until now: {}/{}".format(matched_by_topic, idx+1))
        self.P("")
      doc_prc = doc_acc / len(gt_tags)
      tags_per_doc.append(len(gt_tags))
      if doc_prc == 0:
        zero_preds = True
        doc_prc = zero_penalty
      docs_acc.append(doc_prc)
    overall_acc = np.mean(docs_acc)
    topic_acc = matched_by_topic / len(lst_docs)
    self.last_test_non_zero = not zero_preds
    if record_trace:
      self.train_recall_history.append(round(overall_acc * 100, 1))
      self.train_recall_history_epochs.append(self.train_epoch)
      self.train_topic_recall_history.append(round(topic_acc * 100, 1))
    if show:
      self.P("Tagger benchmark on {} documents with {:.1f} avg tags/doc".format(
          len(lst_docs), np.mean(tags_per_doc)))
      self.P("  {}".format("ZERO PREDS :( !!!" if zero_preds else "Hurray! All preds non-zero! :)))"))
      self.P("  Overall recall: {:5.1f}%".format(overall_acc * 100))
      self.P("  Max doc recall: {:5.1f}%".format(np.max(docs_acc) * 100))
      self.P("  Min doc recall: {:5.1f}%".format(np.min(docs_acc) * 100))
      self.P("  Med doc recall: {:5.1f}%".format(np.median(docs_acc) * 100))
      self.P("  Topic recall  : {:5.1f}%".format(topic_acc * 100))
    return max(0, round(overall_acc * 100, 2)), max(0, round(topic_acc * 100, 2))
  
  def find_topic(self, dict_tags, choose_by_length=False):
    """
    Returns the topic id based on the tags found by the document tagger.
    
    First constructs a topic_identification map: a dictionary where all the keys
    are the topics, and the values are lists of the tags which appear in 
    the documents where the topic appears.
    
    Following that, another dictionary is constructed where the keys are the topics 
    but the values are length of the list in the first dictionary/ sum of the probabilities.
    
    Returns the max value of the dictionary as the found topic
    """
    assert self.dic_topic2tags is not None
    topic_labels = list(self.dic_topic2tags.keys())
    topic_identification_map = {k:list() for k in topic_labels}
    #topic identification on best tags:
    for tag, conf in dict_tags.items():
      for topic in topic_labels:
        if tag in self.dic_topic2tags[topic]:
          topic_identification_map[topic].append((tag,conf))
  
    if choose_by_length:
      topic_identification_len_map = {k: len(v) for k,v in topic_identification_map.items()}
      max_len_key = max(topic_identification_len_map, key=lambda k: topic_identification_len_map[k])
      return max_len_key
    
    else:
      topic_identification_sum_map = {k:0 for k in topic_identification_map.keys()}
    
      for key, values in topic_identification_map.items():
        topic_identification_sum_map[key] += sum([pair[1] for pair in values])
      
      max_sum_key = max(topic_identification_sum_map, key=topic_identification_sum_map.get)
    
      return max_sum_key

  
if __name__ == '__main__':
  from libraries import Logger
  
  cfg1 = "tagger/brain/configs/config.txt"
  
  use_raw_text = True
  force_batch = True
  use_model_conversion = False
  epochs = 30
  use_loaded = True
  
  l = Logger(lib_name="ALNT",config_file=cfg1)
  

  
  eng = ALLANTaggerEngine(log=l,)
  
  eng.initialize()
  
    
  l.P("")
  l.P("")
  tags, inputs = eng.predict_text("as vrea info despre salarizare daca se poate")
  res = eng.tagdict_to_text(tags)
  l.P("Result: {}".format(res))
  l.P(" Debug results: {}".format(['{}:{:.2f}'.format(x,p) 
        for x,p in zip(eng.last_labels, eng.last_probas)]))
      

  l.P("")
  l.P("")
  tags, inputs = eng.predict_text("Aveti cumva sediu si in cluj?")
  res = eng.tagdict_to_text(tags)
  l.P("Result: {}".format(res))
  l.P(" Debug results: {}".format(['{}:{:.2f}'.format(x,p) 
        for x,p in zip(eng.last_labels, eng.last_probas)]))

  l.P("")
  l.P("")
  tags, inputs = eng.predict_text("unde aveti birourile in bucuresti?")
  res = eng.tagdict_to_text(tags)
  l.P("Result: {}".format(res))
  l.P(" Debug results: {}".format(['{}:{:.2f}'.format(x,p) 
        for x,p in zip(eng.last_labels, eng.last_probas)]))


  l.P("")
  l.P("")
  tags, inputs = eng.predict_text("care este atmosfera de echipa in EY?")
  res = eng.tagdict_to_text(tags)
  l.P("Result: {}".format(res))
  l.P(" Debug results: {}".format(['{}:{:.2f}'.format(x,p) 
        for x,p in zip(eng.last_labels, eng.last_probas)]))


  l.P("")
  l.P("")
  tags, inputs = eng.predict_text("in ce zona aveti biroul in Iasi?")
  res = eng.tagdict_to_text(tags)
  l.P("Result: {}".format(res))
  l.P(" Debug results: {}".format(['{}:{:.2f}'.format(x,p) 
        for x,p in zip(eng.last_labels, eng.last_probas)]))

