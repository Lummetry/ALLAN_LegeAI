# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 18:08:35 2022

@author: filip
"""

import numpy as np

from libraries.model_server_v2 import FlaskWorker
from tagger.brain.emb_aproximator import SimpleEmbeddingApproximatorWrapper

import constants as ct

_CONFIG = {
  'TAGGER_MODEL': '20211206_205159_ep35_R0.61_P0.90_F10.73.h5',
  'LABEL2ID': 'dict_lbl_37.pkl',
  'EMBGEN_MODEL' : '20211125_203842_embgen_model_sc_40_ep140.h5',
  'GENERATED_EMBEDS' : 'embgen_full_embeds.npy',
  'WORD_EMBEDS' : 'lai_embeddings_191K.pkl',
  'IDX2WORD' : 'lai_ro_i2w_191K.pkl'
  }

MIN_SUBDOCUMENTS = 2
MIN_SUBDOCUMENT_WORDS = 400
MAX_QUERY_WORDS = 50

MAX_COS_DISTANCE = 0.5


__VER__='0.1.0.0'
class GetMarkWorker(FlaskWorker):
  def __init__(self, **kwargs):
    super(GetMarkWorker, self).__init__(**kwargs)
    return
  
  def _load_model(self):
    fn_tagger_model = self.config_worker['TAGGER_MODEL']
    fn_model = self.config_worker['EMBGEN_MODEL']
    fn_gen_emb = self.config_worker['GENERATED_EMBEDS']
    fn_emb = self.config_worker['WORD_EMBEDS']
    fn_i2w = self.config_worker['IDX2WORD']
    fn_label_to_id = self.config_worker['LABEL2ID']

    self.label_to_id = self.log.load_pickle_from_data(fn_label_to_id)
    self.id_to_label = {v: k for k, v in self.label_to_id.items()}
    self.tagger_model = self.log.load_keras_model(fn_tagger_model)

    self.encoder = SimpleEmbeddingApproximatorWrapper(
      log=self.log,
      fn_embeds=fn_emb,
      fn_idx2word=fn_i2w,
      embgen_model_file=fn_model,
      generated_embeds_filename=fn_gen_emb,
    )

    warmup_input = self.encoder.encode_convert_unknown_words(
      "Warmup",
      fixed_len=ct.MODELS.TAG_MAX_LEN
    )
    self.tagger_model(warmup_input)

    self._create_notification('LOAD', 'Loaded EmbeddingApproximator')
    return


#######
# AUX #
#######

  def cosine_distance(self, a, b):
      a_norm = np.linalg.norm(a)
      b_norm = np.linalg.norm(b)
      similiarity = np.dot(a, b.T)/(a_norm * b_norm)
      dist = 1. - similiarity
      
      return dist

    

  def _pre_process(self, inputs): 
      
    # Read query
    query = inputs['QUERY']
    if len(query.split(' ')) > MAX_QUERY_WORDS:
        raise ValueError("Query: '{}' exceedes max number of allowed words of {}".format(
            query, MAX_QUERY_WORDS))
    # Embed query
    query_embeds = self.encoder.encode_convert_unknown_words(
              query,
              fixed_len=ct.MODELS.TAG_MAX_LEN
    )    
    
    # Read subdocument list
    docs = inputs['DOCUMENTS']    
    if len(docs) < MIN_SUBDOCUMENTS:
      raise ValueError("Number of subdocuments is below the minimum of {}".format(
        MIN_SUBDOCUMENTS))
    
    # Embed each subdocument
    docs_embeds = []    
    for doc in docs:
        if len(doc) < MIN_SUBDOCUMENT_WORDS:
          raise ValueError("Document: '{}' is below the minimum of {} words".format(
            doc, MIN_SUBDOCUMENT_WORDS))
        doc_embeds = self.encoder.encode_convert_unknown_words(
          doc,
          fixed_len=ct.MODELS.TAG_MAX_LEN
        )
        docs_embeds.append(doc_embeds)
        
        
    n_hits = inputs.get('TOP_N', None)

    return query_embeds, docs_embeds, n_hits    


  def _predict(self, prep_inputs):
    query_embeds, docs_embeds, n_hits = prep_inputs
    
    query_tag_vector = self.tagger_model(query_embeds).numpy().squeeze()
    
    doc_distances = []
    
    # Get document distances
    for i, doc_embeds in enumerate(docs_embeds):
        doc_tag_vector = self.tagger_model(doc_embeds).numpy().squeeze()
        distance = self.cosine_distance(query_tag_vector, doc_tag_vector)
        print(i+1, distance)
        
        doc_distances.append((i+1, distance))
        
    # Sort documents by distance
    doc_distances.sort(key=lambda tup: tup[1])
    
    if not n_hits is None:
        # Select the TOP N documents        
        n_hits = int(n_hits)
        if n_hits == 0:
            # Select all
            n_hits = len(doc_distances)            
        selected_docs = [idx for (idx, _) in doc_distances[:n_hits]]
    else:
        # Select using threshold
        selected_docs = [idx for (idx, dist) in doc_distances if dist < MAX_COS_DISTANCE]
    
    return selected_docs

  def _post_process(self, pred):
    selected_docs = pred

    res = {}
    res['results'] = selected_docs

    return res
    
    
if __name__ == '__main__':
  from libraries import Logger
  log = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
  w = GetMarkWorker(log=log, default_config=_CONFIG, verbosity_level=1)

  inputs_to_test = [
    {
      'QUERY' : 'Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti?',
      'DOCUMENTS': [
          """Subsemnatul Damian Ionut Andrei, domiciliat in Cluj, Strada Cernauti, nr. 17-21, bl. J, parter, ap. 1 , nascut pe data 24-01-1982, declar pe propria raspundere ca sotia mea Andreea Damian, avand domiciliul flotant in Bucuresti, str. Drumul Potcoavei nr 120, bl. B, sc. B, et. 1, ap 5B, avand CI cu CNP 1760126413223 serie RK, numar 897567 nu detine averi ilicite""",
          """decizia recurată a fost dată cu încălcarea autorităţii de lucru interpretat, respectiv cu încălcarea dispozitivului hotărârii preliminare pronunţate de Curtea de Justiţie a Uniunii Europene în Cauza C-52/07 (hotărâre care are autoritate de lucru interpretat „erga omnes”)""",
            """Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti? Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti? Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti?""",
          ]
    },
    
    # {
    #   'QUERY' : 'Operaţiunea întocmirii referatului de necesitate se situează în interiorul procedurii de achiziţie publică?',
    #   'TOP_N' : 0,
    #   'DOCUMENTS': [
    #       """Etapa de planificare/pregătire a unui proces de achiziţie publică se iniţiază prin identificarea necesităţilor şi elaborarea referatelor de necesitate şi se încheie cu aprobarea de către conducătorul autorităţii contractante/unităţii de achiziţii centralizate a documentaţiei de atribuire, inclusiv a documentelor-suport, precum şi a strategiei de contractare pentru procedura respectivă.""",
    #       """Referatul de necesitate reprezintă un document intern emis de fiecare compartiment din cadrul autorităţii contractante în ultimul trimestru al anului în curs pentru anul viitor, care cuprinde necesităţile de produse, servicii şi lucrări identificate, precum şi preţul unitar/total al necesităţilor.""",
    #       """documentaţia de atribuire - documentul achiziţiei care cuprinde cerinţele, criteriile, regulile şi alte informaţii necesare pentru a asigura operatorilor economici o informare completă, corectă şi explicită cu privire la cerinţe sau elemente ale achiziţiei, obiectul contractului şi modul de desfăşurare a procedurii de atribuire, inclusiv specificaţiile tehnice ori documentul descriptiv, condiţiile contractuale propuse, formatele de prezentare a documentelor de către candidaţi/ofertanţi, informaţiile privind obligaţiile generale aplicabile;""",
    #       """Etapa de organizare a procedurii de atribuire a contractului/acordului-cadru începe prin transmiterea documentaţiei de atribuire în SEAP şi se finalizează odată cu încheierea contractului de achiziţie publică/acordului-cadru.""",
    #       """(1) Autoritatea contractantă poate iniţia aplicarea procedurii de atribuire numai după ce au fost elaborate şi aprobate documentaţia de atribuire şi documentele suport.""",
    #       """(1) Procedurile de atribuire reglementate de prezenta lege, aplicabile pentru atribuirea contractelor de achiziţie publică/acordurilor-cadru sau organizarea concursurilor de soluţii cu o valoare estimată egală sau mai mare decât valorile prevăzute la art. 7 alin. (5), sunt următoarele:"""
    #       ]
    # },
    
    # {
    #   'QUERY' : 'Cum se concretizează sprijinul acordat, conform legii, de celelalte compartimente interne din cadrul entității contractante, compartimentului intern specializat în domeniul achizițiilor, în funcție de specificul și complexitatea obiectului achiziției?',
    #   'TOP_N' : 0,
    #   'DOCUMENTS': [
    #       """a) transmiterea referatelor de necesitate care cuprind necesităţile de produse, servicii şi lucrări, valoarea estimată a acestora, precum şi informaţiile de care dispun, potrivit competenţelor, necesare pentru elaborarea strategiei de contractare a respectivelor contracte sectoriale/acorduri-cadru;""",
    #       """b) transmiterea, dacă este cazul, a specificaţiilor tehnice aşa cum sunt acestea prevăzute la art. 165 din Lege;""",
    #       """c) în funcţie de natura şi complexitatea necesităţilor identificate în referatele prevăzute la lit. a), transmiterea de informaţii cu privire la preţul unitar/total actualizat al respectivelor necesităţi, în urma unei cercetări a pieţei sau pe bază istorică;""",
    #       """d) informarea cu privire la fondurile alocate pentru fiecare destinaţie, precum şi poziţia bugetară a acestora;""",
    #       """e) informarea justificată cu privire la eventualele modificări intervenite în execuţia contractelor sectoriale/acordurilor-cadru, care cuprinde cauza, motivele şi oportunitatea modificărilor propuse;""",
    #       # """f) transmiterea documentului constatator privind modul de îndeplinire a clauzelor contractuale.""",
    #       """a) întreprinde demersurile necesare pentru înregistrarea/reînnoirea/recuperarea înregistrării entităţii contractante în SEAP sau recuperarea certificatului digital, dacă este cazul;""",
    #       """b) elaborează şi, după caz, actualizează, pe baza necesităţilor transmise de celelalte compartimente ale entităţii contractante, programul anual al achiziţiilor sectoriale şi, dacă este cazul, strategia anuală de achiziţii;""",          
    #       """c) elaborează sau, după caz, coordonează activitatea de elaborare a documentaţiei de atribuire şi a strategiei de contractare sau, în cazul organizării unui concurs de soluţii, a documentaţiei de concurs şi a strategiei de contractare, pe baza necesităţilor transmise de compartimentele de specialitate;""",
    #       """h) verificarea propunerilor financiare prezentate de ofertanţi, inclusiv verificarea conformităţii cu propunerile tehnice, verificarea aritmetică, verificarea încadrării în fondurile care pot fi disponibilizate pentru îndeplinirea contractului sectorial respectiv, precum şi, dacă este cazul, verificarea încadrării acestora în situaţia prevăzută la art. 222 din Lege;""",
    #       """i) elaborarea solicitărilor de clarificări şi/sau completări necesare în vederea evaluării solicitărilor de participare şi/sau ofertelor;""",
    #       """j) stabilirea solicitărilor de participare neadecvate, a ofertelor inacceptabile şi/sau neconforme, precum şi a motivelor care stau la baza încadrării acestora în fiecare din aceste categorii;""",
    #       """l) aplicarea criteriului de atribuire şi a factorilor de evaluare, astfel cum a fost prevăzut în anunţul de participare/simplificat/de concurs;""",
    #       ]
    # },
  ]

  for i,_input in enumerate(inputs_to_test):
    result = w.execute(inputs=_input, counter=i)
    print(result)
