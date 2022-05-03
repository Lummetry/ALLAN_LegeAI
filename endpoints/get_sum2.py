# -*- coding: utf-8 -*-

from libraries.model_server_v2 import FlaskWorker
from tagger.brain.emb_aproximator import SimpleEmbeddingApproximatorWrapper

import constants as ct
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import spacy
from gensim.models import Word2Vec
import tensorflow as tf

_CONFIG = {
  'LABEL2ID': 'dict_lbl_37.pkl',
  'EMBGEN_MODEL': '20211125_203842_embgen_model_sc_40_ep140.h5',
  'GENERATED_EMBEDS': 'embgen_full_embeds.npy',
  'WORD_EMBEDS': 'lai_embeddings_191K.pkl',
  'IDX2WORD': 'lai_ro_i2w_191K.pkl',
  'MODEL' : '20211124_082955_e128_v191K_final',
  'SPACY_MODEL' : 'ro_core_news_md',
 }


# File paths

# Debug
# ENCODER_DEBUG = 'C:\\Proiecte\\LegeAI\\Date\\Task5\\models\\ohe_vocab10_e10_d15_l256_enc.h5'
# DECODER_DEBUG = 'C:\\Proiecte\\LegeAI\\Date\\Task5\\models\\ohe_vocab10_e10_d15_l256_dec.h5'
# DICT_ID2WORD_DEBUG = 'C:\\Proiecte\\LegeAI\\Date\\Task5\\models\\dictId2Word_min10.pkl'
ENCODER_DEBUG = 'C:\\Proiecte\\LegeAI\\Date\\Task5\\models\\att_vocab5_e10_d15_l256_enc.h5'
DECODER_DEBUG = 'C:\\Proiecte\\LegeAI\\Date\\Task5\\models\\att_vocab5_e10_d15_l256_dec.h5'
DICT_ID2WORD_DEBUG = 'C:\\Proiecte\\LegeAI\\Date\\Task5\\models\\dictId2Word_min5.pkl'
SHUFFLER_DEBUG = 'C:\\Proiecte\\LegeAI\\Date\\Task5\\models\\shuffle_decoder_newproc.h5'

# Prod
ENCODER_PROD = 'C:\\allan_data\\2022.04.19\\sum_enc.h5'
DECODER_PROD = 'C:\\allan_data\\2022.04.19\\sum_dec.h5'
DICT_ID2WORD_PROD = 'C:\\allan_data\\2022.04.19\\dictId2Word.pkl'
SHUFFLER_PROD = 'C:\\allan_data\\2022.04.19\\shuffle_decoder_newproc.h5'

EMBEDDING_DIM = 128 
MAX_OUTPUT_LEN = 10
ENCODER_SEQ_DIM = 10
DECODER_SEQ_DIM = 15

__VER__='2.2.1.0'
class GetSum2Worker(FlaskWorker):
    """
    Implementation of the worker for GET_SUMMARY endpoint
    """
    
    
    def __init__(self, **kwargs):
      super(GetSum2Worker, self).__init__(**kwargs)
      return

    def _load_model(self):
        fn_model = self.config_worker['EMBGEN_MODEL']
        fn_gen_emb = self.config_worker['GENERATED_EMBEDS']
        fn_emb = self.config_worker['WORD_EMBEDS']
        fn_i2w = self.config_worker['IDX2WORD']
        fn_label_to_id = self.config_worker['LABEL2ID']
    
        self.label_to_id = self.log.load_pickle_from_data(fn_label_to_id)
        self.id_to_label = {v: k for k, v in self.label_to_id.items()}
        
        self.encoder = SimpleEmbeddingApproximatorWrapper(
          log=self.log,
          fn_embeds=fn_emb,
          fn_idx2word=fn_i2w,
          embgen_model_file=fn_model,
          generated_embeds_filename=fn_gen_emb,
        )
        self._create_notification('LOAD', 'Loaded EmbeddingApproximator')
        
        # Word2Vec
        model_fn = self.log.get_models_file(self.config_worker['MODEL'])
        self.model = Word2Vec.load(model_fn)
        self._create_notification('LOAD', 'Loaded model {}'.format(model_fn))
                
        # Load Romanian spaCy dataset
        try:
            self.nlp_model = spacy.load(self.config_worker['SPACY_MODEL'])
        except OSError:
            spacy.cli.download(self.config_worker['SPACY_MODEL'])
            self.nlp_model = spacy.load(self.config_worker['SPACY_MODEL'])         
                  
        # Set the distance function
        self.dist_func = self.encoder.encoder._setup_dist_func(func_name='cos')
    
        return
    
    
    def _decode_doc(self, doc):
        '''
        Replace Romanian special characters in a document.

        '''
        original_chars = ['Î', 'î', 'ă', 'Ă', 'â', 'â', 'Ș', 'ș', 'Ț', 'ț', 'ş', 'ţ', '„', '”']
        replace_chars = ['I', 'i', 'a', 'A', 'a', 'a', 'S', 's', 'T', 't', 's', 't', '"', '"']
        
        for i in range(len(original_chars)):
            doc = doc.replace(original_chars[i], replace_chars[i])
            
        return doc
    
    
    def _select_cluster_words(self,
                             cluster_words, word_embed_distances,
                             tfs=None, idfs=None,
                             n_selected=None,
                             debug=False
                            ):
        ''' Select words from cluster:
            - sort words by their score
            - eliminate more distant words which are included in others
            - select the closest words
        '''
                    
        # Sort words    
        if tfs is None:
            word_scores = word_embed_distances
            zip_list = zip(cluster_words, word_scores)
            reverse = False
            
        else:
            # If TF and IDF were provided for the words
            word_scores = [t * i * (1/(d + 0.00001)) for (t, i, d) in zip(tfs, idfs, word_embed_distances)]
            zip_list = zip(cluster_words, word_scores, tfs, idfs, word_embed_distances)
            reverse = True
        
        zip_list = sorted(zip_list, key=lambda tup: tup[1], reverse=reverse)
        
        
        # Eliminate words which are contained in others
        selected_zip_list = []
        for i, pair1 in enumerate(reversed(zip_list)):
            word1 = pair1[0]
            
            contained = False
            for pair2 in reversed(zip_list[:-(i+1)]):
                word2 = pair2[0]
                
                if word1.startswith(word2) or word2.startswith(word1):
                    # One of the words is contained in the other
                    contained = True
                    break
                    
            if not contained:
                selected_zip_list.append(pair1)
        selected_zip_list.reverse()
        
        # Select the closest words
        if n_selected:
            selected_zip_list = selected_zip_list[:n_selected]
        
        cluster_words = [t[0] for t in selected_zip_list]
        word_scores = [t[1] for t in selected_zip_list]
        
        return cluster_words, word_scores
    
    
    def getWordFromOHE(self, ohe):
        """ Get the word from an OHE array """
        
        idx = np.argmax(ohe)
        word = self.dict_id2word.get(idx, '<UNK>')
        
        return word
    
    def getWordEmbedding(self, word):
        """ Get the encoding for a word. """
        
        emb = self.encoder.encode_convert_unknown_words(
            [word],
            fixed_len=EMBEDDING_DIM
        ) 
        
        return emb[0][0]
    
    def decodeIdSeq(self, seq):
        """ Decode a sequence of ids into a sequence of words. """
        
        words = []
        for idx in seq:
            word = self.dict_id2word.get(idx)
            if word != '<unk>':
                words.append(word)
        
        sentence = ' '.join(words)
        return sentence
    
    
    def decode_sequenceOHE(self, encoder_model, decoder_model, input_seq):
        """ Inference loop for the Seq2Seq model with OHE output. """    
        
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)
    
        # Generate empty target sequence of length 1.
        target_seq = np.copy(self.start_token)
        target_seq.resize(1, 1, EMBEDDING_DIM)
        
        res_seq = []    
        stop_condition = False
        
        stopIdx = np.argmax(self.stop_ohe)
        unkIdx = self.vocab_length - 1
        
        i = 0
        
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        while not stop_condition:
            
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
    
            # Sample a token
            idx = np.argmax(output_tokens[0, -1, :])
    
            # Exit condition: either hit max length, find stop token or repeat last word
            if idx == stopIdx or i > MAX_OUTPUT_LEN or (i > 1 and idx != unkIdx and idx == res_seq[-1]):
                stop_condition = True
            else:
                i += 1 
                res_seq.append(idx)
    
                # Update the target sequence (of length 1).
                word = self.getWordFromOHE(output_tokens[0, -1, :])
                target_seq = self.getWordEmbedding(word)
                target_seq = target_seq.reshape([1, 1, EMBEDDING_DIM])
    
                # Update states
                states_value = [h, c]
            
        res_seq = np.array(res_seq)
        
        decodedRes = self.decodeIdSeq(res_seq)
            
        return decodedRes

    def decode_sequenceOHEAttention(self, encoder_model, decoder_model, input_seq):
        """ Inference loop for the Seq2Seq model with Attention and OHE output. """    
        
        # Encode the input as state vectors.
        encoder_states, encoder_outputs = encoder_model.predict(input_seq)
    
        # Generate empty target sequence of length 1.
        target_seq = np.copy(self.start_token)
        target_seq.resize(1, 1, EMBEDDING_DIM)
        
        res_seq = []    
        stop_condition = False
        
        stopIdx = np.argmax(self.stop_ohe)
        unkIdx = self.vocab_length - 1
        
        i = 0
        
        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        while not stop_condition:
            
            output_tokens, h, c = decoder_model.predict([target_seq, encoder_outputs, encoder_states])
    
            # Sample a token
            idx = np.argmax(output_tokens[0, -1, :])
    
            # Exit condition: either hit max length, find stop token or repeat last word
            if idx == stopIdx or i > MAX_OUTPUT_LEN or (i > 1 and idx != unkIdx and idx == res_seq[-1]):
                print(i)
                stop_condition = True
            else:
                i += 1 
                res_seq.append(idx)
    
                # Update the target sequence (of length 1).
                word = self.getWordFromOHE(output_tokens[0, -1, :])
                target_seq = self.getWordEmbedding(word)
                target_seq = target_seq.reshape([1, 1, EMBEDDING_DIM])
    
                # Update states
                encoder_states = [h, c]
            
        res_seq = np.array(res_seq)
        
        decodedRes = self.decodeIdSeq(res_seq)
            
        return decodedRes
    
    def get_sort_regular_positions(self, pred):
        """ Get the list of word positions for regular output model by sorting the outputs. """
        
        order = np.argsort(pred, axis=None)
        
        result = np.zeros(len(pred), dtype=int)
        for o, idx in enumerate(order):
            result[idx] = o
            
        return result
    
    
    def build_ordered_sentence(self, order, words):
        """ Form the final sentence, according to the predicted order of words. """
        
        ordered_words = []
        
        for idx in order:
            if idx < len(words) and words[idx] != '<PAD>':
                # If the word index exists and is not <PAD>
                ordered_words.append(words[idx])
        
        sentence = ' '.join(ordered_words)
        
        return sentence
    
    def order_first_appearance(self, text, selected_words):
        """ Order the list of selected words by their first appearence in the text. """
        
        ordered_words = []
        
        for word in text:
            try:
                idx = selected_words.index(word)
                ordered_words.append(word)
                del selected_words[idx]
            except ValueError:
                continue
            
        sentence = ' '.join(ordered_words)
        
        return sentence
    
    def try_next(self, current_idx, word_distances, visited_idxs):
        # Recursive function to find the best path from all the neighbors
        
        min_path_length = np.Infinity
        best_visited_idxs = None
                
        if all(visited_idxs) > 0:
            # If all indexes have been visited, stop
            return 0, visited_idxs
        
        
        for idx in range(len(visited_idxs)):
            if word_distances[current_idx, idx] > 0 and visited_idxs[idx] == 0:
                # Visit new neighbor index
                
                # Update copy of visited indexes
                new_visited_idxs = np.copy(visited_idxs)
                new_visited_idxs[idx] = max(new_visited_idxs) + 1
                
                # Get the best path through the visited index
                returned_path_length, returned_visited_idxs = self.try_next(idx, word_distances, new_visited_idxs)
                path_length = word_distances[current_idx, idx] + returned_path_length
                
                if path_length < min_path_length:
                    # Update the best path for the current index
                    best_visited_idxs = returned_visited_idxs
                    min_path_length = path_length 
        
        return min_path_length, best_visited_idxs
    
    
    def order_shortest_path(self, text, selected_words):
        """ Order the selected words by the shortest path the covers all of them, according to their
            positions in the text.
        """
    
        # Calculate word distance graph
    
        n = len(selected_words)
        word_distances = np.zeros((n, n))
    
        for i in range(len(text) - 1):
            try:
                idx1 = selected_words.index(text[i])
                pos1 = i
            
                # Only consider distances of at most 15 words
                end = int(min(len(text), i + 16))
                    
                for j in range(i + 1, end):
                    # Check the net words in the text
                    try:
                        idx2 = selected_words.index(text[j])
                        pos2 = j
                        
                        if idx1 != idx2:
                            # If a different word was found
                            if word_distances[idx1, idx2] == 0 or word_distances[idx1, idx2] > pos2 - pos1:
                                # If the distance between the words is better than the previous
                                word_distances[idx1, idx2] = pos2 - pos1  
                                
                    except ValueError:
                        continue
                        
            except ValueError:
                continue
    
        # Add a fake word / padding as the start token
        pad_word_distances = np.insert(word_distances, 0, [np.ones(len(selected_words))], axis=0)
        pad_word_distances = np.insert(pad_word_distances, 0, [np.ones(len(selected_words) + 1)], axis=1)
        pad_word_distances[0, 0] = 0
        
        # Calculate shortest covering path
    
        # Add one element for the padding        
        visited_idxs = np.zeros(len(selected_words) + 1, dtype=int)
    
        visited_idxs[0] = 1
        length, order = self.try_next(0, pad_word_distances, visited_idxs)
        
        if order is None:
            return ''
    
        # Ignore the start pad
        order = list(order[1:] - 2)
    
        ordered_words = [w for _, w in sorted(zip(order, selected_words))]
    
        sentence = ' '.join(ordered_words)
        
        return sentence
    
    
    def _pre_process(self, inputs):
        
        self.debug = bool(inputs.get('DEBUG', False))
        
        # Read files
        if self.debug:
            # encoder_model_file = ENCODER_DEBUG
            # decoder_model_file = DECODER_DEBUG
            # dict_id2word_file = DICT_ID2WORD_DEBUG
            shuffle_model_file = SHUFFLER_DEBUG
        else:
            # encoder_model_file = ENCODER_PROD
            # decoder_model_file = DECODER_PROD
            # dict_id2word_file = DICT_ID2WORD_PROD
            shuffle_model_file = SHUFFLER_PROD
            
        # self.encoder_model = tf.keras.models.load_model(encoder_model_file)
        # self.decoder_model = tf.keras.models.load_model(decoder_model_file) 
        self.shuffle_model = tf.keras.models.load_model(shuffle_model_file) 
        
        # dict_id2word_file = open(dict_id2word_file, "rb")
        # self.dict_id2word = pickle.load(dict_id2word_file)
        # dict_id2word_file.close() 
        # self.vocab_length = len(self.dict_id2word)     
        
        # Start and Stop tokens
        # self.start_token = np.zeros(EMBEDDING_DIM)
        # self.stop_ohe = np.zeros(self.vocab_length).astype(int)
        # self.stop_ohe[-2] = 1
        
                            
        doc = inputs['DOCUMENT']
        if len(doc) < ct.MODELS.TAG_MIN_INPUT:
          raise ValueError("Document: '{}' is below the minimum of {} words".format(
            doc, ct.MODELS.TAG_MIN_INPUT))          
        
        nlp_doc = self.nlp_model(doc)
        # Keep only nouns and verb lemmas
        lemmas = []
        for token in nlp_doc:
            
            # Only accept nouns and verbs and lemmas with more the 2 letters
            if token.pos_ in ['NOUN', 'VERB'] and len(token.lemma_) > 2:
                
                # Replace Romanian special characters
                lemma = self._decode_doc(token.lemma_).lower()
                
                lemmas.append(lemma)
        
        # Get the embeddings for the words
        embeds = self.encoder.encode_convert_unknown_words(
            [lemmas],
            fixed_len=0
        )[0]             
        
        n_hits = 0        
        multi_cluster = True
        
        lemmas = np.array(lemmas)
    
        return embeds, lemmas, n_hits, multi_cluster

    def _predict(self, prep_inputs):
        
        embeds, words, n_hits, multi_cluster = prep_inputs
        
        # Term frequency
        unique_words, word_frequency = np.unique(words, return_counts=True)
        frequency_dict = dict(zip(unique_words, word_frequency))
        
        # Agglomerative clustering
        cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6, 
                                          affinity='cosine', linkage='average')  
        cluster.fit_predict(embeds)  
        
        # Get top clusters by size
        cluster_labels, cluster_sizes = np.unique(cluster.labels_, return_counts=True)
        
        if multi_cluster:
            min_cluster_size = np.mean(cluster_sizes[cluster_sizes > 1])
        else:
            min_cluster_size = np.min(cluster_sizes[cluster_sizes > 1])
            
        cluster_labels = cluster_labels[cluster_sizes >= min_cluster_size]
        cluster_sizes = cluster_sizes[cluster_sizes >= min_cluster_size]        
        
        top_clusters = cluster_labels[np.argsort(-cluster_sizes)]
        
        # Analyze each of the top clusters
        selected_words = []
        for c in top_clusters:
            
            # Select members of cluster
            idxs = np.nonzero(cluster.labels_ == c)[0] 
            cluster_words = words[idxs]
            cluster_embeds = embeds[idxs]
            
            n = len(cluster_embeds)
            cluster_center = np.sum(cluster_embeds, axis=0) / n
            
            # Get distances between the center of the cluster and all its members            
            word_embed_distances = self.dist_func(cluster_embeds, cluster_center)
                        
            if multi_cluster:
                n_selected = int(n / min_cluster_size)
            else:
                n_selected = 1
                
            # Calculate TF and IDF
            tfs = []
            idfs = []
            for word in cluster_words:
                tf = np.log(1 + frequency_dict[word])
                
                try:
                    idf = np.log(1 + (self.model.corpus_count / self.model.wv.get_vecattr(word, "count")))
                except:
                    idf = 1
                
                tfs.append(tf)
                idfs.append(idf)
            
            # Select words from the cluster
            cluster_selected_words, cluster_selected_scores = self._select_cluster_words(cluster_words,
                                                                                        word_embed_distances,
                                                                                        tfs=tfs, idfs=idfs,
                                                                                        n_selected=n_selected,
                                                                                        debug=self.debug
                                                                                        )                
            
            selected_words.extend(cluster_selected_words)
            
        
        # Seq2Seq Model
        
        # Get embeddings for tags
        # tag_embeds = self.encoder.encode_convert_unknown_words(
        #     [selected_words[:(ENCODER_SEQ_DIM)]],
        #     fixed_len=ENCODER_SEQ_DIM
        # )  
        # decoded_sentence = self.decode_sequenceOHE(self.encoder_model, self.decoder_model, tag_embeds)
        # decoded_sentence = self.decode_sequenceOHEAttention(self.encoder_model, self.decoder_model, tag_embeds)
        
        # V1 Shuffle model
        
        # Get embeddings for tags
        tag_embeds = self.encoder.encode_convert_unknown_words(
            [selected_words[:(MAX_OUTPUT_LEN)]],
            fixed_len=MAX_OUTPUT_LEN
        )  
                
        # Get the predicted order
        pred = self.shuffle_model.predict(tag_embeds)
        order = self.get_sort_regular_positions(pred[0])
        decoded_sentence_v1 = self.build_ordered_sentence(order, selected_words[:(MAX_OUTPUT_LEN)])
        
        # V2 Text order
        decoded_sentence_v2 = self.order_first_appearance(words, selected_words[:(MAX_OUTPUT_LEN)])
        
        # V3 Shortest text path
        decoded_sentence_v3 = self.order_shortest_path(words, selected_words[:(MAX_OUTPUT_LEN)])
        
              
        return selected_words, decoded_sentence_v1, decoded_sentence_v2, decoded_sentence_v3

    def _post_process(self, pred):
        
        words, sentence_v1, sentence_v2, sentence_v3 = pred
        
        res = {}
        
        res['v0'] = words
        res['v1'] = sentence_v1
        res['v2'] = sentence_v2
        res['v3'] = sentence_v3
        
        return res


if __name__ == '__main__':
  from libraries import Logger

  l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
  eng = GetSum2Worker(log=l, default_config=_CONFIG, verbosity_level=1)
  
  test = {
        # 'DOCUMENT': 'Aceasta situatie ridica semne de intrebare privind consolidarea bugetara, potrivit constructiei bugetare initiale. Desi proiectul de buget tintește un deficit cash de din PIB, apreciaza ca nu sunt suficiente masuri credibile de ajustare bugetara care sa conduca la atingerea acestei tinte, se arata in document.',
      
#         'DOCUMENT' : """Extinderea sistemului Uniunii Europene de taxare a emisiilor de CO2 pentru ca acesta să fie aplicat din anul 2026 încălzirii clădirilor şi transporturilor rutiere de mărfuri provoacă diviziuni între statele UE, împărţite în două blocuri de această chestiune inclusă de Comisia Europeană în pachetul de măsuri pentru atingerea obiectivelor climatice ale UE. 
# Astfel, o parte din ţările membre consideră că aceasta ar fi o măsură eficientă de diminuare a emisiilor de carbon generate de activitatea economică, iar cealaltă parte se teme de o creştere a costurilor pentru populaţie care ar produce un impact social de genul „vestelor galbene” în Franţa, transmite luni agenţia EFE.
# În al doilea grup de ţări se regăsesc state est-europene, dar şi altele precum Grecia sau Spania, ele însumând circa o treime din populaţia UE. La Consiliul UE pentru mediu, desfăşurat luni la Bruxelles, aceste ţări s-au arătat sceptice sau ostile iniţiativei care ar conduce la o creştere costurilor pentru încălzire şi transporturi, una dintre măsurile din planul Comisiei Europene ce are ca obiectiv reducerea cu cel puţin 55% a emisiilor de CO2 ale UE până în anul 2030, faţă de nivelurile anului 1990.
# „Trebuie să analizăm mai mult impactul” acestei măsuri, a declarat ministrul spaniol al tranziţiei ecologice, Teresa Ribera. Aceasta consideră important ca cetăţenii să perceapă „oportunităţile” tranziţiei ecologice, un lucru „complicat” în actuala conjunctură a preţurilor la energie, care cresc facturile atât pentru populaţie, cât şi pentru industrie. 
# Totuşi, Teresa Ribera a subliniat că Spania sprijină „ambiţia ecologică şi proiectul economic care inspiră” pachetul de iniţiative menite să reducă emisiile de CO2 ale economiei europene, pe care l-a descris drept „pachetul legislativ cel mai important al deceniului”, notează Agerpres.
# Spania s-a delimitat astfel de ţările est-europene care critică propunerea de extindere a sistemului european de comercializare a emisiilor de carbon (ETS) şi în general planul Comisiei Europene de reducere a emisiilor de CO2 denumit 'Fit for 55', precum Cehia, Polonia sau Ungaria. Acestea din urmă consideră că respectivul plan va face ca mai degrabă populaţia, şi nu poluatorii, să suporte costurile combaterii schimbărilor climatice.
# În favoarea taxării emisiilor de CO2 generate de transporturi şi încălzirea locuinţelor s-au declarat în special ţări din centrul şi nordul Europei, precum Austria, Germania, Olanda, Suedia, Danemarca sau Finlanda, conform cărora această iniţiativă este „fundamentală” pentru decarbonizarea economiei.
# Între cele două blocuri de ţări comunitare divizate de această chestiune s-a format şi un al treilea, cu o poziţionare mai ambiguă, fiind aici regăsite Luxemburgul, Belgia sau Italia. Cea din urmă a sugerat să fie dezbătute în paralel obiectivele reducerii emisiilor de CO2 pentru fiecare ţară membră în parte şi așa-numita taxa vamală pe CO2 propusă în acelaşi pachet, respectiv taxarea emisiilor de CO2 generate de fabricarea unor produse importate din state terţe.
# Printre măsurile propuse de executivul comunitar în vederea atingerii obiectivului de reducere a emisiilor de gaze cu efect de seră cu 55% faţă de anul 1990 şi obţinerea neutralităţii climatice în anul 2050 se regăsesc: 
# creşterea costurilor pentru emisiile de CO2 generate de încălzire, transporturi şi industrie (inclusiv prin reformarea sistemului UE de comercializare a certificatelor de emisii - ETS);
# taxarea combustibililor destinaţi transporturilor aeriene şi navale;
# interzicerea din anul 2035 a comercializării autoturismelor noi diesel şi pe benzină;
# utilizarea sporită a surselor regenerabile de energie (cu obiectivul ca 40 % din energia UE să fie produsă din surse regenerabile până în 2030) concomitent cu reducerea consumului de energie;
# alinierea politicilor fiscale la obiectivele Pactului Verde European sau realizarea de campanii de împădurire.
# Comisia Europeană a subliniat că acest plan va implica o tranziţie profundă, cu mari schimbări structurale în foarte puţin timp şi va conduce la transformarea economiei şi a societăţii UE în vederea atingerii obiectivelor ambiţioase în materie de climă.""",

# Impozit pe profit
#         'DOCUMENT' : """Impozit pe profit
# Principalele modificări din perspectiva impozitului pe profit prevăzute de Ordonanță constau în:
# Se introduc clarificări cu privire perioada impozabilă pentru persoanele juridice străine care au
# locul de exercitare a conducerii și care se înregistrează în cursul unui an fiscal început.
# Se completează condițiile de aplicare a prevederilor Directivei 2011/96/UE a Consiliului din 30
# noiembrie 2011 privind regimul fiscal comun care se aplică societăților-mamă și filialelor
# acestora din diferite state membre, prin introducerea mențiunii „sau un alt impozit care
# substituie unul dintre acele impozite”.
# Se modifică formularea uneia dintre condiții de a fi considerată „filială dintr-un stat membru”
# prevăzute de către art. 24, în sensul înlocuirii mențiunii existente la condiția prevăzută la
# punctul 4 de a plăti în conformitate cu legislația fiscală a unui stat membru, fără posibilitatea
# unei opțiuni sau exceptări, unul dintre impozitele prevăzute în anexa nr. 2 care face parte
# integrantă din prezentul titlu sau un impozit similar impozitului pe profit reglementat de titlul de
# impozit pe profit cu mențiunea de mai sus (sau un alt impozit care substituie impozitul pe
# profit/unul dintre aceste impozite).
# Se modifică limita de deductibilitate pentru ajustările pentru deprecierea creanțelor (dacă sunt
# îndeplinite condițiile) de la 30% la 50%, începând cu data de 1 ianuarie 2022.
# În ceea ce privește regimul fiscal de impozitare la ieșire, prin derogare de la prevederile art.
# 184 alin. (1) din Codul de procedură fiscală, contribuabilul care aplică regulile de la alin. (1)-(3)
# beneficiază de dreptul de eșalonare la plată pentru acest impozit, prin achitarea în rate egale
# pe parcursul a cinci ani, dacă se află în oricare dintre anumite situații specificate în Codul fiscal.
# De asemenea, se introduc și condiții detaliate în care se acordă dreptul la eșalonare mai sus
# menționat.
# Se modifică art. 43 din Codul fiscal cu privire la declararea, reținerea și plata impozitului pe
# dividende prin eliminarea referinței la situațiile financiare anuale, în contextul în care
# dividendele pot fi distribuite și în alte perioade contabile (de ex. trimestrial), prin înlocuirea
# sintagmei „până la sfârșitul anului în care s-au aprobat situațiile financiare anuale” în „până la
# sfârșitul anului în care s-a aprobat distribuirea acestora”.
# Se introduc două noi reguli specifice la art. 45 din Codul fiscal:
# o Pentru contribuabilii care aplică sistemul anticipat de declarare și plată a impozitului pe
# profit și care vor beneficia de Ordonanța de urgență a Guvernului nr. 153/2020 pentru
# instituirea unor măsuri fiscale de stimulare a menținerii/creșterii capitalurilor proprii –
# aceștia vor efectua plata anticipată pentru trimestrul I al fiecărui an fiscal/an fiscal modificat
# la nivelul sumei rezultate din aplicarea cotei de impozit asupra profitului contabil al
# perioadei pentru care se efectuează plata anticipată, până la data de 25 inclusiv a lunii
# următoare trimestrului I.
# o Sumele care se scad din impozitul pe profit anual (prevăzute la art. I alin. (12) lit. a) din
# Ordonanța de urgență a Guvernului nr. 153/2020) se completează cu „alte sume care se
# scad din impozitul pe profit, potrivit legislației în vigoare”.""",

#Articolul 6 - Dreptul la libertate și la siguranță
#         'DOCUMENT' : """Drepturile prevăzute la articolul 6 corespund drepturilor garantate la articolul 5 din CEDO şi au, în conformitate cu articolul 52 alineatul (3) din cartă, acelaşi înţeles şi acelaşi domeniu de aplicare. Prin urmare, restrângerile la care pot fi supuse în mod legal nu le pot depăşi pe cele permise de articolul 5 din CEDO, care este redactat după cum urmează:
# `(1) Orice persoană are dreptul la libertate şi la siguranţă. Nimeni nu poate fi lipsit de libertatea sa, cu excepţia următoarelor cazuri şi potrivit căilor legale:
# a) dacă este deţinut legal pe baza condamnării de către un tribunal competent;
# b) dacă a făcut obiectul unei arestări sau al unei deţineri legale pentru nesupunerea la o hotărâre pronunţată, conform legii, de către un tribunal ori în vederea garantării executării unei obligaţii prevăzute de lege;
# c) dacă a fost arestat sau reţinut în vederea aducerii sale în faţa autorităţii judiciare competente, atunci când există motive verosimile de a bănui că a săvârşit o infracţiune sau când există motive temeinice de a crede în necesitatea de a-l împiedica să săvârşească o infracţiune sau să fugă după săvârşirea acesteia;
# d) dacă este vorba de detenţia legală a unui minor, hotărâtă pentru educaţia sa sub supraveghere, sau despre detenţia sa legală, în scopul aducerii sale în faţa autorităţii competente;
# e) dacă este vorba despre detenţia legală a unei persoane susceptibile să transmită o boală contagioasă, a unui alienat, a unui alcoolic, a unui toxicoman sau a unui vagabond;
# f) dacă este vorba despre arestarea sau detenţia legală a unei persoane pentru a o împiedica să pătrundă în mod ilegal pe un teritoriu sau împotriva căreia se află în curs o procedură de expulzare ori de extrădare.
# (2) Orice persoană arestată trebuie să fie informată, în termenul cel mai scurt şi într-o limbă pe care o înţelege, asupra motivelor arestării sale şi asupra oricărei acuzaţii aduse împotriva sa.
# (3) Orice persoană arestată sau deţinută în condiţiile prevăzute la alineatul (1) litera (c) din prezentul articol trebuie adusă de îndată înaintea unui judecător sau a altui magistrat împuternicit prin lege cu exercitarea atribuţiilor judiciare şi are dreptul de a fi judecată într-un termen rezonabil sau eliberată în cursul procedurii. Punerea în libertate poate fi subordonată unei garanţii care să asigure prezentarea persoanei în cauză la audiere.
# (4) Orice persoană privată de libertatea sa prin arestare sau deţinere are dreptul de a introduce un recurs în faţa unui tribunal, pentru ca acesta să statueze într-un termen scurt asupra legalităţii deţinerii sale şi să dispună eliberarea sa dacă deţinerea este ilegală.
# (5) Orice persoană care este victima unei arestări sau a unei deţineri în condiţii contrare dispoziţiilor acestui articol are dreptul la reparaţii.`
# Drepturile prevăzute la articolul 6 trebuie respectate în special la adoptarea de către Parlamentul European şi de către Consiliu a actelor legislative în domeniul cooperării judiciare în materie penală, în temeiul articolelor 82, 83 şi 85 din Tratatul privind funcţionarea Uniunii Europene, mai ales pentru a defini dispoziţiile comune minime privind calificarea infracţiunilor şi pedepsele, precum şi anumite aspecte de drept procedural.""",

# Definiţia sediului permanent
#         'DOCUMENT' : """În înţelesul prezentului cod, sediul permanent este un loc prin care se desfăşoară integral sau parţial activitatea unui nerezident, fie direct, fie printr-un agent dependent.
# (2) Un sediu permanent presupune un loc de conducere, sucursală, birou, fabrică, magazin, atelier, precum şi o mină, un puţ de ţiţei sau gaze, o carieră sau alte locuri de extracţie a resurselor naturale.
# (3) Un sediu permanent presupune un şantier de construcţii, un proiect de construcţie, ansamblu sau montaj sau activităţi de supervizare legate de acestea, numai dacă şantierul, proiectul sau activităţile durează mai mult de 6 luni.
# (4) Prin derogare de la prevederile alin. (1)-(3), un sediu permanent nu presupune următoarele:
# a) folosirea unei instalaţii numai în scopul depozitării sau al expunerii produselor ori bunurilor ce aparţin nerezidentului;
# b) menţinerea unui stoc de produse sau bunuri ce aparţin unui nerezident numai în scopul de a fi depozitate sau expuse;
# c) menţinerea unui stoc de produse sau bunuri ce aparţin unui nerezident numai în scopul de a fi procesate de către o altă persoană;
# d) vânzarea de produse sau bunuri ce aparţin unui nerezident, care au fost expuse în cadrul unor expoziţii sau târguri fără caracter permanent ori ocazionale, dacă produsele ori bunurile sunt vândute nu mai târziu de o lună după încheierea târgului sau a expoziţiei;
# e) păstrarea unui loc fix de activitate numai în scopul achiziţionării de produse sau bunuri ori culegerii de informaţii pentru un nerezident;
# f) păstrarea unui loc fix de activitate numai în scopul desfăşurării de activităţi cu caracter pregătitor sau auxiliar de către un nerezident;
# g) păstrarea unui loc fix de activitate numai pentru o combinaţie a activităţilor prevăzute la lit. a)-f), cu condiţia ca întreaga activitate desfăşurată în locul fix să fie de natură preparatorie sau auxiliară.
# (5) Prin derogare de la prevederile alin. (1) şi (2), un nerezident este considerat a avea un sediu permanent în România, în ceea ce priveşte activităţile pe care o persoană, alta decât un agent cu statut independent, le întreprinde în numele nerezidentului, dacă persoana acţionează în România în numele nerezidentului şi dacă este îndeplinită una din următoarele condiţii:
# a) persoana este autorizată şi exercită în România autoritatea de a încheia contracte în numele nerezidentului, cu excepţia cazurilor în care activităţile respective sunt limitate la cele prevăzute la alin. (4) lit. a)-f);
# b) persoana menţine în România un stoc de produse sau bunuri din care livrează produse sau bunuri în numele nerezidentului.
# (6) Un nerezident nu se consideră că are un sediu permanent în România dacă doar desfăşoară activitate în România prin intermediul unui broker, agent, comisionar general sau al unui agent intermediar având un statut independent, în cazul în care această activitate este activitatea obişnuită a agentului, conform descrierii din documentele constitutive. Dacă activităţile unui astfel de agent sunt desfăşurate integral sau aproape integral în numele nerezidentului, iar în relaţiile comerciale şi financiare dintre nerezident şi agent există condiţii diferite de acelea care ar exista între persoane independente, agentul nu se consideră ca fiind agent cu statut independent.
# (7) Un nerezident nu se consideră că are un sediu permanent în România numai dacă acesta controlează sau este controlat de un rezident ori de o persoană ce desfăşoară o activitate în România prin intermediul unui sediu permanent sau altfel.
# (8) În înţelesul prezentului cod, sediul permanent al unei persoane fizice se consideră a fi baza fixă.""",


        'DOCUMENT' : """Atribuţiile de stare civilă se îndeplinesc în cadrul:
a) serviciilor publice comunitare judeţene de evidenţă a persoanelor, respectiv al Direcţiei Generale de Evidenţă a Persoanelor a municipiului Bucureşti, denumită în continuare D.G.E.P.M.B.;
b) serviciilor publice comunitare locale de evidenţă a persoanelor;
c) primăriilor unităţilor administrativ-teritoriale unde nu sunt constituite servicii publice comunitare locale de evidenţă a persoanelor;
d) misiunilor diplomatice şi oficiilor consulare de carieră ale României.
(2) Sunt ofiţeri de stare civilă:
a) la nivelul unităţilor administrativ-teritoriale din cadrul judeţelor: primarii municipiilor, oraşelor şi comunelor, precum şi persoanele care au atribuţii de stare civilă;
b) la nivelul municipiului Bucureşti: primarul general al municipiului Bucureşti, primarii sectoarelor şi persoanele care au atribuţii de stare civilă din cadrul sectoarelor municipiului Bucureşti;
c) la nivelul misiunilor diplomatice şi al oficiilor consulare de carieră ale României: şefii acestora.
(3) Sunt ofiţeri de stare civilă cu atribuţii restrânse:
a) comandanţii de nave şi aeronave;
b) ofiţerii de stare civilă desemnaţi prin ordin al ministrului apărării naţionale sau, după caz, al ministrului afacerilor interne, potrivit prevederilor art. 7 alin. (7).
(4) Activităţile de stare civilă se realizează de către:
a) funcţionarii cu atribuţii de stare civilă din cadrul instituţiilor prevăzute la alin. (1) lit. a);
b) ofiţerii de stare civilă din cadrul instituţiilor prevăzute la alin. (1) lit. b) - d).
(5) Primarii şi şefii misiunilor diplomatice şi ai oficiilor consulare de carieră ale României pot delega sau retrage, după caz, exercitarea atribuţiilor de ofiţer de stare civilă viceprimarului sau secretarului unităţii administrativ-teritoriale ori altor funcţionari publici din aparatul propriu, respectiv diplomatului care îndeplineşte funcţii consulare ori unuia dintre funcţionarii consulari.
(6) Prin excepţie de la prevederile alin. (5), şefii misiunilor diplomatice şi ai oficiilor consulare de carieră ale României pot delega sau retrage, după caz, exercitarea atribuţiilor de ofiţer de stare civilă personalului angajat al Ministerului Afacerilor Externe trimis în misiune permanentă sau temporară pe funcţii tehnico-administrative, care prestează activitate consulară, în situaţiile în care diplomatul care îndeplineşte funcţii consulare sau funcţionarul consular se află în imposibilitatea legală de a întocmi actul de stare civilă.
(7) Ofiţerul de stare civilă nu poate întocmi acte de stare civilă când este parte sau declarant.""",


    
        # 'TOP_N': 0,
        
        # 'MULTI_CLUSTER': True,
        
        'DEBUG': True
      }
  
  res = eng.execute(inputs=test, counter=1)
  print(res)
  