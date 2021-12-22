# -*- coding: utf-8 -*-

from libraries.model_server_v2 import FlaskWorker
from tagger.brain.emb_aproximator import SimpleEmbeddingApproximatorWrapper

import constants as ct
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import spacy

_CONFIG = {
  'LABEL2ID': 'dict_lbl_37.pkl',
  'EMBGEN_MODEL': '20211125_203842_embgen_model_sc_40_ep140.h5',
  'GENERATED_EMBEDS': 'embgen_full_embeds.npy',
  'WORD_EMBEDS': 'lai_embeddings_191K.pkl',
  'IDX2WORD': 'lai_ro_i2w_191K.pkl'
 }

class GetSumWorker(FlaskWorker):
    """
    Implementation of the worker for GET_SUMMARY endpoint
    """
    
    
    def __init__(self, **kwargs):
      super(GetSumWorker, self).__init__(**kwargs)
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
        
        # Load Romanian spaCy dataset
        self.nlp_model = spacy.load('ro_core_news_md')        
                  
        # Set the distance function
        self.dist_func = self.encoder.encoder._setup_dist_func(func_name='cos')
    
        self._create_notification('LOAD', 'Loaded EmbeddingApproximator')
        return
    
    
    def _decode_doc(self, doc):
        '''
        Replace Romanian special characters in a document.

        '''
        original_chars = ['Î', 'î', 'ă', 'Ă', 'â' 'Ș', 'ș', 'Ț', 'ț', 'ş', 'ţ', '„', '”']
        replace_chars = ['I', 'i', 'a', 'A', 'a' 'S', 's', 'T', 't', 's', 't', '"', '"']
        
        for i in range(len(original_chars)):
            doc = doc.replace(original_chars[i], replace_chars[i])
            
        return doc


    def _pre_process(self, inputs):
                
        doc = inputs['DOCUMENT']
        if len(doc) < ct.MODELS.TAG_MIN_INPUT:
          raise ValueError("Document: '{}' is below the minimum of {} words".format(
            doc, ct.MODELS.TAG_MIN_INPUT))
          
        
        nlp_doc = self.nlp_model(doc)
        # Keep only nouns and verb lemmas
        lemmas = []
        for token in nlp_doc:
            if token.pos_ in ['NOUN', 'VERB']:
                
                # Replace Romanian special characters
                lemma = self._decode_doc(token.lemma_)
                
                lemmas.append(lemma)
        
        # Get the embeddings for the words
        embeds = self.encoder.encode_convert_unknown_words(
            [lemmas],
            fixed_len=0
        )[0]             
        
        n_hits = int(inputs.get('TOP_N', 1))
    
        return embeds, lemmas, n_hits

    def _predict(self, prep_inputs):
        
        embeds, words, n_hits = prep_inputs
        
        cluster = AgglomerativeClustering(n_clusters=None, distance_threshold=0.6, 
                                          affinity='cosine', linkage='average')  
        cluster.fit_predict(embeds)
        
        # print(cluster.labels_)
        
        # Get top clusters by size
        cluster_labels, cluster_sizes = np.unique(cluster.labels_, return_counts=True)
        top_clusters = np.unique(cluster_labels[np.argsort(-cluster_sizes)[:n_hits]])
        
        selected_words = []
        for c in top_clusters:
            cluster_sum = np.zeros(len(embeds[0]))
            n = 0
            
            for i, e in enumerate(embeds):
                if cluster.labels_[i] == c:
                    cluster_sum += e
                    n += 1
            
            # print('Cluster {} - {} elements.'.format(c, n))
            
            cluster_sum = cluster_sum / n
            # print(self.encoder.encoder.decode([[embeds[0]]], tokens_as_embeddings=True))
            
            # idx = self.encoder.encoder._get_closest_idx(cluster_sum, top=5)
            # for ix in idx:
            #     print(self.encoder.encoder.dic_index2word.get(ix))
                        
            word_embed_distances = self.encoder.encoder.dist(target=cluster_sum, source=embeds)
            id_min = np.argmin(word_embed_distances)
            selected_words.append(words[id_min])
            
              
        return selected_words

    def _post_process(self, pred):
        
        res = {}
        res['results'] = pred
        
        return res


if __name__ == '__main__':
  from libraries import Logger

  l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
  eng = GetSumWorker(log=l, default_config=_CONFIG, verbosity_level=1)
  
  test = {
        # 'DOCUMENT': 'Aceasta situatie ridica semne de intrebare privind consolidarea bugetara, potrivit constructiei bugetare initiale. Desi proiectul de buget tintește un deficit cash de din PIB, apreciaza ca nu sunt suficiente masuri credibile de ajustare bugetara care sa conduca la atingerea acestei tinte, se arata in document.',
      
        'DOCUMENT' : """Extinderea sistemului Uniunii Europene de taxare a emisiilor de CO2 pentru ca acesta să fie aplicat din anul 2026 încălzirii clădirilor şi transporturilor rutiere de mărfuri provoacă diviziuni între statele UE, împărţite în două blocuri de această chestiune inclusă de Comisia Europeană în pachetul de măsuri pentru atingerea obiectivelor climatice ale UE. 
Astfel, o parte din ţările membre consideră că aceasta ar fi o măsură eficientă de diminuare a emisiilor de carbon generate de activitatea economică, iar cealaltă parte se teme de o creştere a costurilor pentru populaţie care ar produce un impact social de genul „vestelor galbene” în Franţa, transmite luni agenţia EFE.
În al doilea grup de ţări se regăsesc state est-europene, dar şi altele precum Grecia sau Spania, ele însumând circa o treime din populaţia UE. La Consiliul UE pentru mediu, desfăşurat luni la Bruxelles, aceste ţări s-au arătat sceptice sau ostile iniţiativei care ar conduce la o creştere costurilor pentru încălzire şi transporturi, una dintre măsurile din planul Comisiei Europene ce are ca obiectiv reducerea cu cel puţin 55% a emisiilor de CO2 ale UE până în anul 2030, faţă de nivelurile anului 1990.
„Trebuie să analizăm mai mult impactul” acestei măsuri, a declarat ministrul spaniol al tranziţiei ecologice, Teresa Ribera. Aceasta consideră important ca cetăţenii să perceapă „oportunităţile” tranziţiei ecologice, un lucru „complicat” în actuala conjunctură a preţurilor la energie, care cresc facturile atât pentru populaţie, cât şi pentru industrie. 
Totuşi, Teresa Ribera a subliniat că Spania sprijină „ambiţia ecologică şi proiectul economic care inspiră” pachetul de iniţiative menite să reducă emisiile de CO2 ale economiei europene, pe care l-a descris drept „pachetul legislativ cel mai important al deceniului”, notează Agerpres.
Spania s-a delimitat astfel de ţările est-europene care critică propunerea de extindere a sistemului european de comercializare a emisiilor de carbon (ETS) şi în general planul Comisiei Europene de reducere a emisiilor de CO2 denumit 'Fit for 55', precum Cehia, Polonia sau Ungaria. Acestea din urmă consideră că respectivul plan va face ca mai degrabă populaţia, şi nu poluatorii, să suporte costurile combaterii schimbărilor climatice.
În favoarea taxării emisiilor de CO2 generate de transporturi şi încălzirea locuinţelor s-au declarat în special ţări din centrul şi nordul Europei, precum Austria, Germania, Olanda, Suedia, Danemarca sau Finlanda, conform cărora această iniţiativă este „fundamentală” pentru decarbonizarea economiei.
Între cele două blocuri de ţări comunitare divizate de această chestiune s-a format şi un al treilea, cu o poziţionare mai ambiguă, fiind aici regăsite Luxemburgul, Belgia sau Italia. Cea din urmă a sugerat să fie dezbătute în paralel obiectivele reducerii emisiilor de CO2 pentru fiecare ţară membră în parte şi așa-numita taxa vamală pe CO2 propusă în acelaşi pachet, respectiv taxarea emisiilor de CO2 generate de fabricarea unor produse importate din state terţe.
Printre măsurile propuse de executivul comunitar în vederea atingerii obiectivului de reducere a emisiilor de gaze cu efect de seră cu 55% faţă de anul 1990 şi obţinerea neutralităţii climatice în anul 2050 se regăsesc: 
creşterea costurilor pentru emisiile de CO2 generate de încălzire, transporturi şi industrie (inclusiv prin reformarea sistemului UE de comercializare a certificatelor de emisii - ETS);
taxarea combustibililor destinaţi transporturilor aeriene şi navale;
interzicerea din anul 2035 a comercializării autoturismelor noi diesel şi pe benzină;
utilizarea sporită a surselor regenerabile de energie (cu obiectivul ca 40 % din energia UE să fie produsă din surse regenerabile până în 2030) concomitent cu reducerea consumului de energie;
alinierea politicilor fiscale la obiectivele Pactului Verde European sau realizarea de campanii de împădurire.
Comisia Europeană a subliniat că acest plan va implica o tranziţie profundă, cu mari schimbări structurale în foarte puţin timp şi va conduce la transformarea economiei şi a societăţii UE în vederea atingerii obiectivelor ambiţioase în materie de climă.""",
      
        'TOP_N': 5
      }
  
  res = eng.execute(inputs=test, counter=1)
  print(res)
