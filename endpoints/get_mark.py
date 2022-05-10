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
MIN_SUBDOCUMENT_WORDS = 1
MAX_QUERY_WORDS = 350

MAX_COS_DISTANCE = 0.5


__VER__='0.1.1.3'
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
  
    self.debug = bool(inputs.get('DEBUG', False))
      
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
    for i, doc in enumerate(docs):
        num_words = len(doc.split(' '))
        
        if self.debug:
            print('Doc {} - {} words.'.format(i+1, num_words))
        
        if num_words < MIN_SUBDOCUMENT_WORDS:
           raise ValueError("Document: '{}' is below the minimum of {} words".format(
             doc, MIN_SUBDOCUMENT_WORDS))
           
        doc_embeds = self.encoder.encode_convert_unknown_words(
          doc,
          fixed_len=ct.MODELS.TAG_MAX_LEN
        )
        docs_embeds.append(doc_embeds)
        
        
    n_hits = inputs.get('TOP_N', 3)

    return query_embeds, docs_embeds, n_hits    


  def _predict(self, prep_inputs):
    query_embeds, docs_embeds, n_hits = prep_inputs
    
    query_tag_vector = self.tagger_model(query_embeds).numpy().squeeze()
    
    doc_distances = []
    
    # Get document distances
    for i, doc_embeds in enumerate(docs_embeds):
        doc_tag_vector = self.tagger_model(doc_embeds).numpy().squeeze()
        distance = self.cosine_distance(query_tag_vector, doc_tag_vector)
        
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
    # {
    #   'QUERY' : 'Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti?',
    #   'DOCUMENTS': [
    #       # """Subsemnatul Damian Ionut Andrei, domiciliat in Cluj, Strada Cernauti, nr. 17-21, bl. J, parter, ap. 1 , nascut pe data 24-01-1982, declar pe propria raspundere ca sotia mea Andreea Damian, avand domiciliul flotant in Bucuresti, str. Drumul Potcoavei nr 120, bl. B, sc. B, et. 1, ap 5B, avand CI cu CNP 1760126413223 serie RK, numar 897567 nu detine averi ilicite""",
    #       """decizia recurată a fost dată cu încălcarea autorităţii de lucru interpretat, respectiv cu încălcarea dispozitivului hotărârii preliminare pronunţate de Curtea de Justiţie a Uniunii Europene în Cauza C-52/07 (hotărâre care are autoritate de lucru interpretat „erga omnes”) decizia recurată a fost dată cu încălcarea autorităţii de lucru interpretat, respectiv cu încălcarea dispozitivului hotărârii preliminare pronunţate de Curtea de Justiţie a Uniunii Europene în Cauza C-52/07 (hotărâre care are autoritate de lucru interpretat „erga omnes”) decizia recurată a fost dată cu încălcarea autorităţii de lucru interpretat, respectiv cu încălcarea dispozitivului hotărârii preliminare pronunţate de Curtea de Justiţie a Uniunii Europene în Cauza C-52/07 (hotărâre care are autoritate de lucru interpretat „erga omnes”)""",
    #         """Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti? Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti? Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti? Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti? Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti? Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti? Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti? Care este tva-ul intracomunitar ce se aplica atunci cand aduci masini SH de la nemti?""",
    #       ]
    # },
    
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
    #       ],
    #   'DEBUG' : True,
    # },
    
    # {
    #   'QUERY' : 'Cum se concretizează sprijinul acordat, conform legii, de celelalte compartimente interne din cadrul entității contractante, compartimentului intern specializat în domeniul achizițiilor, în funcție de specificul și complexitatea obiectului achiziției?',
    #   'TOP_N' : 0,
    #   'DOCUMENTS': [
    #       "a) transmiterea referatelor de necesitate care cuprind necesităţile de produse, servicii şi lucrări, valoarea estimată a acestora, precum şi informaţiile de care dispun, potrivit competenţelor, necesare pentru elaborarea strategiei de contractare a respectivelor contracte sectoriale/acorduri-cadru;",
    #       "b) transmiterea, dacă este cazul, a specificaţiilor tehnice aşa cum sunt acestea prevăzute la art. 165 din Lege;",
    #       "c) în funcţie de natura şi complexitatea necesităţilor identificate în referatele prevăzute la lit. a), transmiterea de informaţii cu privire la preţul unitar/total actualizat al respectivelor necesităţi, în urma unei cercetări a pieţei sau pe bază istorică;",
    #       "d) informarea cu privire la fondurile alocate pentru fiecare destinaţie, precum şi poziţia bugetară a acestora;",
    #       "e) informarea justificată cu privire la eventualele modificări intervenite în execuţia contractelor sectoriale/acordurilor-cadru, care cuprinde cauza, motivele şi oportunitatea modificărilor propuse;",
    #       "f) transmiterea documentului constatator privind modul de îndeplinire a clauzelor contractuale.",
    #       "a) întreprinde demersurile necesare pentru înregistrarea/reînnoirea/recuperarea înregistrării entităţii contractante în SEAP sau recuperarea certificatului digital, dacă este cazul;",
    #       "b) elaborează şi, după caz, actualizează, pe baza necesităţilor transmise de celelalte compartimente ale entităţii contractante, programul anual al achiziţiilor sectoriale şi, dacă este cazul, strategia anuală de achiziţii;",
    #       "c) elaborează sau, după caz, coordonează activitatea de elaborare a documentaţiei de atribuire şi a strategiei de contractare sau, în cazul organizării unui concurs de soluţii, a documentaţiei de concurs şi a strategiei de contractare, pe baza necesităţilor transmise de compartimentele de specialitate;",
    #       "d) îndeplineşte obligaţiile referitoare la publicitate, astfel cum sunt acestea prevăzute de Lege;",
    #       "e) aplică şi finalizează procedurile;",
    #       "f) realizează achiziţiile directe;",
    #       "g) constituie şi păstrează dosarul achiziţiei sectoriale.",
    #       "(4) Celelalte compartimente ale entităţii contractante au obligaţia de a sprijini activitatea compartimentului intern specializat în domeniul achiziţiilor sectoriale, în funcţie de specificul şi complexitatea obiectului achiziţiei.",
    #       "c) tipul de contract propus şi modalitatea de implementare a acestuia;",
    #       "d) mecanismele de plată în cadrul contractului, alocarea riscurilor în cadrul acestuia, măsuri de gestionare a acestora, stabilirea penalităţilor pentru neîndeplinirea sau îndeplinirea defectuoasă a obligaţiilor contractuale;",
    #       "a) deschiderea ofertelor şi, după caz, a altor documente care însoţesc oferta;",
    #       "b) verificarea îndeplinirii criteriilor de calificare de către ofertanţi/candidaţi;",
    #       "c) realizarea selecţiei candidaţilor, dacă este cazul;",
    #       "d) desfăşurarea dialogului cu operatorii economici, în cazul aplicării procedurii de dialog competitiv;",
    #       "e) desfăşurarea negocierilor cu operatorii economici, în cazul aplicării procedurilor de negociere;",
    #       "f) verificarea conformităţii propunerilor tehnice ale ofertanţilor cu prevederile caietului de sarcini;",
    #       "g) evaluarea propunerilor tehnice ale ofertanţilor în conformitate cu criteriile de atribuire, dacă este cazul;",
    #       "h) verificarea propunerilor financiare prezentate de ofertanţi, inclusiv verificarea conformităţii cu propunerile tehnice, verificarea aritmetică, verificarea încadrării în fondurile care pot fi disponibilizate pentru îndeplinirea contractului sectorial respectiv, precum şi, dacă este cazul, verificarea încadrării acestora în situaţia prevăzută la art. 222 din Lege;",
    #       "i) elaborarea solicitărilor de clarificări şi/sau completări necesare în vederea evaluării solicitărilor de participare şi/sau ofertelor;",
    #       "j) stabilirea solicitărilor de participare neadecvate, a ofertelor inacceptabile şi/sau neconforme, precum şi a motivelor care stau la baza încadrării acestora în fiecare din aceste categorii;",
    #       "k) stabilirea ofertelor admisibile;",
    #       "l) aplicarea criteriului de atribuire şi a factorilor de evaluare, astfel cum a fost prevăzut în anunţul de participare/simplificat/de concurs;",
    #       "m) stabilirea ofertei/ofertelor câştigătoare sau, după caz, formularea propunerii de anulare a procedurii;",
    #       "n) elaborarea proceselor-verbale aferente fiecărei şedinţe, a rapoartelor intermediare aferente fiecărei etape în cazul procedurilor cu mai multe etape şi a raportului procedurii de atribuire.",
    #       "(2) Rapoartele intermediare şi raportul procedurii de atribuire se înaintează de către preşedintele comisiei de evaluare conducătorului entităţii contractante spre aprobare.",
    #       "p) contractul sectorial/acordul-cadru, semnate, şi, după caz, actele adiţionale;"
    #       """a) transmiterea referatelor de necesitate care cuprind necesităţile de produse, servicii şi lucrări, valoarea estimată a acestora, precum şi informaţiile de care dispun, potrivit competenţelor, necesare pentru elaborarea strategiei de contractare a respectivelor contracte sectoriale/acorduri-cadru;""",
    #       """b) transmiterea, dacă este cazul, a specificaţiilor tehnice aşa cum sunt acestea prevăzute la art. 165 din Lege;""",
    #       """c) în funcţie de natura şi complexitatea necesităţilor identificate în referatele prevăzute la lit. a), transmiterea de informaţii cu privire la preţul unitar/total actualizat al respectivelor necesităţi, în urma unei cercetări a pieţei sau pe bază istorică;""",
    #       """d) informarea cu privire la fondurile alocate pentru fiecare destinaţie, precum şi poziţia bugetară a acestora;""",
    #       """e) informarea justificată cu privire la eventualele modificări intervenite în execuţia contractelor sectoriale/acordurilor-cadru, care cuprinde cauza, motivele şi oportunitatea modificărilor propuse;""",
    #       """f) transmiterea documentului constatator privind modul de îndeplinire a clauzelor contractuale.""",
    #       """a) întreprinde demersurile necesare pentru înregistrarea/reînnoirea/recuperarea înregistrării entităţii contractante în SEAP sau recuperarea certificatului digital, dacă este cazul;""",
    #       """b) elaborează şi, după caz, actualizează, pe baza necesităţilor transmise de celelalte compartimente ale entităţii contractante, programul anual al achiziţiilor sectoriale şi, dacă este cazul, strategia anuală de achiziţii;""",          
    #       """c) elaborează sau, după caz, coordonează activitatea de elaborare a documentaţiei de atribuire şi a strategiei de contractare sau, în cazul organizării unui concurs de soluţii, a documentaţiei de concurs şi a strategiei de contractare, pe baza necesităţilor transmise de compartimentele de specialitate;""",
    #       """h) verificarea propunerilor financiare prezentate de ofertanţi, inclusiv verificarea conformităţii cu propunerile tehnice, verificarea aritmetică, verificarea încadrării în fondurile care pot fi disponibilizate pentru îndeplinirea contractului sectorial respectiv, precum şi, dacă este cazul, verificarea încadrării acestora în situaţia prevăzută la art. 222 din Lege;""",
    #       """i) elaborarea solicitărilor de clarificări şi/sau completări necesare în vederea evaluării solicitărilor de participare şi/sau ofertelor;""",
    #       """j) stabilirea solicitărilor de participare neadecvate, a ofertelor inacceptabile şi/sau neconforme, precum şi a motivelor care stau la baza încadrării acestora în fiecare din aceste categorii;""",
    #       """l) aplicarea criteriului de atribuire şi a factorilor de evaluare, astfel cum a fost prevăzut în anunţul de participare/simplificat/de concurs;""",
    #       ],
    #          'DEBUG' : True,
    # },
    
    # {
    #   'QUERY' : 'Întocmirea și asumarea caietului de sarcini se realizează de către compartimentul intern specializat în domeniul achiziţiilor publice sau de către compartimentul intern beneficiar al achiziției publice?',
    #   'TOP_N' : 0,
    #   'DOCUMENTS': [
    #       "In vederea realizării achiziţiilor publice, autoritatea contractantă înfiinţează în condiţiile legii un compartiment intern specializat în domeniul achiziţiilor, format, de regulă, din minimum trei persoane, dintre care cel puţin două treimi având studii superioare, precum şi specializări în domeniul achiziţiilor.",
    #       "Decizia Consiliului a fost necesară pentru a activa mecanismul instituit de Directiva 2001/55/CE privind standardele minime pentru acordarea protecţiei temporare, în cazul unui aflux masiv de persoane deplasate şi măsurile de promovare a unui echilibru între eforturile statelor membre pentru primirea acestor persoane şi suportarea consecinţelor acestei primiri. ",
    #       "În aplicarea prezentelor norme metodologice, autoritatea contractantă, prin compartimentul intern specializat în domeniul achiziţiilor publice, are următoarele atribuţii principale:  a) întreprinde demersurile necesare pentru înregistrarea/reînnoirea/recuperarea înregistrării autorităţii contractante în SEAP sau recuperarea certificatului digital, dacă este cazul;  b) elaborează şi, după caz, actualizează, pe baza necesităţilor transmise de celelalte compartimente ale autorităţii contractante, programul anual al achiziţiilor publice şi, dacă este cazul, strategia anuală de achiziţii;",
    #       "Prin utilizarea interfețelor electronice, persoana impozabilă - care facilitează pe această cale vânzarea la distanță de bunuri importate din teritorii terțe sau țări terțe în loturi cu o valoare intrinsecă de maximum 150 euro, precum și livrarea de bunuri în UE de către o persoană impozabilă nestabilită în UE către o persoană neimpozabilă - se consideră că a primit și a livrat ea însăși bunurile respective. Faptul generator intervine și TVA devine exigibilă în aceste cazuri în momentul în care plata a fost acceptată",
    #       "atunci când bunurile sunt importate într-un alt stat membru decât cel în care se încheie transportul bunurilor către client, locul este considerat a fi unde se află bunurile în momentul în care se încheie transportul acestora către client; o atunci când bunurile sunt importate în statul membru în care se încheie transportul bunurilor către client, locul este considerat a fi statul membru respectiv, cu condiția ca TVA pentru aceste bunuri să fie d",
    #       "Având în vedere faptul că preluarea activității de soluționare a contestațiilor fiscale reprezintă un proces laborios și de durată, precum și faptul că, până la momentul împlinirii celor 6 luni de la data publicării în Monitorul Oficial a Legii nr. 295/2020 (i.e. 22 iunie 2021), nu au fost adoptate acte normative care să asigure implementarea și organizarea acestui proces, Guvernul României a decis că: ✓ Ministerul Finanțelor Publice, prin structura sa de specialitate, va prelua activitatea de soluționare a contestațiilor formulate împotriva titlurilor de creanță și împotriva altor acte administrativ-fiscale",
    #       ],
    #   'DEBUG' : True,
    # },
    
    # {
    #   'QUERY' : 'Dacă autoritate contractantă solicită, prin caietul de sarcini, o echipă de experţi în domeniul urbanismului, motivând cerinţa în conformitate cu Regulamentul referitor la organizarea şi funcţionarea Registrului Urbaniştilor din România (RUR) și în același timp utilizează drept criteriu de atribuire "cel mai bun raport calitate-preţ", iar unul din factorii de evaluare este experienţa similară a experţilor-cheie, concretizată în numărul de proiecte similare, detaliat în subfactori de evaluare pentru fiecare dintre experţii cheie, sunt aplicabile prevederile art. 12, nota (II) din Instrucţiunea preşedintelui A.N.A.P. nr. 1/2017, în cazul în care autoritatea contractantă solicită prezentarea unor experţi-cheie prin caietul de sarcini şi, simultan, să utilizeze ca factori de evaluare experienţa similară a acestor experţi, concretizată în numărul de proiecte similare?',
    #   'TOP_N' : 0,
    #   'DOCUMENTS': [
    #       "(ii) Experţii precizaţi la art. 3 alin. (3), care prin obţinerea certificării obţin implicit şi competenţele necesare desfăşurării activităţii în cauză, nu pot fi utilizaţi ca factori de evaluare, având în vedere faptul că, odată obţinut un nivel de certificare, se apreciază că rezultatele obţinute în urma prestaţiilor acestora nu pot fi diferite, din punct de vedere calitativ, într-o măsură semnificativă.",
    #       """(1) Sintagma "personalul ce va realiza efectiv activităţile care fac obiectul contractului ce urmează a fi atribuit" se referă la experţii/personalul-cheie ce răspund(e) de realizarea efectivă a proceselor de execuţie aferente implementării respectivului contract, calificarea, experienţa profesională şi/sau modul de organizare influenţând, în mod direct, calitatea rezultatului ce trebuie atins prin contractul în cauză, această categorie de personal nefiind considerată criteriu de calificare şi selecţie în raport cu operatorul economic ce este candidat/ofertant în procedura de atribuire.""",
    #       """(2) Prin noţiunea de "personal-cheie" menţionată la alin. (1), se înţelege orice expert/specialist a cărui activitate desfăşurată în cadrul contractului este reflectată direct fie:""",
    #       """(3) În cadrul acestei categorii se includ şi tipurile de experţi pentru care este impusă, prin legislaţia de specialitate din domeniul obiectului contractului ce urmează a fi atribuit, prezentarea unei certificări specifice, fără de care aceştia nu au dreptul de a exercita activitatea în cauză, în acest fel fiind legaţi indisolubil de implementarea propriu-zisă a respectivului contract.""",
    #       """(4) Pentru tipurile de experţi menţionaţi la alin. (3), pentru care existenţa certificării specifice, emisă de un organism abilitat conform prevederilor legale incidente domeniului în cauză, reprezintă condiţia necesară şi suficientă pentru a putea duce la îndeplinire activităţile ce fac obiectul respectivelor certificări, autoritatea/entitatea contractantă nu va stabili criterii de calificare şi selecţie, ci va solicita ca în propunerea tehnică să fie descris momentul în care vor interveni aceşti experţi în implementarea viitorului contract, precum şi modul în care operatorul economic ofertant şi-a asigurat accesul la serviciile acestora (fie prin resurse proprii, caz în care vor fi prezentate persoanele în cauză, fie prin externalizare, situaţie în care se vor descrie aranjamentele contractuale realizate în vederea obţinerii serviciilor respective).""",
    #       """(5) În sensul alin. (3), în situaţia în care existenţa certificării specifice nu este suficientă pentru a demonstra că respectivul expert are capacitatea de a îndeplini activităţile aferente din viitorul contract, datorită complexităţii ridicate şi/sau particularităţii acestora, autoritatea/entitatea contractantă are dreptul de a impune în caietul de sarcini şi cerinţe referitoare la experienţa profesională a expertului/specialistului în cauză, ce trebuie îndeplinite la momentul evaluării ofertei tehnice.""",
    #       """(2) Propunerea tehnică trebuie să corespundă cerinţelor minime prevăzute în caietul de sarcini sau în documentul descriptiv.""",
    #       """a) nu satisface în mod corespunzător cerinţele caietului de sarcini;""",
    #       """a) factorii de evaluare sunt asociaţi unei/unor extinderi a cerinţei/cerinţelor minime obligatorii stabilite prin caietul de sarcini, avantajul urmărit fiind în corelaţie cu valori superioare ale nivelurilor de calificare şi/sau experienţă profesională prezentate de experţii-cheie, ce fac obiectul factorilor de evaluare, faţă de nivelul minim ce trebuie îndeplinit pentru ca propunerea tehnică să fie declarată conformă;""",
    #       """b) factorii de evaluare sunt relevanţi şi reflectă avantajele economice/financiare rezultate din cadrul ofertelor depuse, avantaje care să nu fie anulate sau diminuate pe parcursul îndeplinirii contractului, fiind corelaţi cu specificul activităţilor şi domeniului corespunzătoare obiectului respectivului contract.""",
    #       """b) stabilirea exercitării dreptului de semnătură în raport cu tipurile de documentaţii de amenajare a teritoriului şi de urbanism;"""

    #       ],
    #   'DEBUG' : True,
    # },
    
    # {
    #   'QUERY' : 'O instituţie publică de interes naţional, finanţată integral de la bugetul de stat care este înregistrată în SEAP ca utilizator "autoritate contractantă" se poate înregistra în SEAP şi ca utilizator "operator economic", în vederea participării la proceduri de achiziţii publice derulate prin SEAP în calitate de ofertant?',
    #   'TOP_N' : 0,
    #   'DOCUMENTS': [
    #       """jj) operator economic - orice persoană fizică sau juridică, de drept public ori de drept privat, sau grup ori asociere de astfel de persoane, inclusiv orice asociere temporară formată între două ori mai multe dintre aceste entităţi, care oferă în mod licit pe piaţă executarea de lucrări, furnizarea de produse ori prestarea de servicii, şi care este/sunt stabilită/stabilite în:""",
    #       """gg) ofertant - orice operator economic care a depus o ofertă în cadrul unei proceduri de atribuire;""",
    #       """(1) Orice autoritate contractantă, precum şi orice operator economic care utilizează SEAP în vederea participării la o procedură de atribuire, are obligaţia de a solicita înregistrarea şi, respectiv, reînnoirea înregistrării în SEAP.""",
    #       """(3) Responsabilitatea pentru corecta funcţionare din punct de vedere tehnic a SEAP revine operatorului acestui sistem.""",
    #       """(1) Înregistrarea, reînnoirea şi recuperarea înregistrării în SEAP se efectuează respectându-se procedura electronică implementată de către operatorul SEAP, cu avizul Agenţiei Naţionale pentru Achiziţii Publice, denumită în continuare ANAP, potrivit atribuţiilor.""",
    #       """(2) Procedura electronică pentru înregistrarea, reînnoirea şi recuperarea înregistrării, atât pentru autorităţile contractante, cât şi pentru operatorii economici, se publică pe site-ul www.e-licitatie.ro""",
    #       """Autorităţile contractante şi operatorii economici care solicită înregistrarea, reînnoirea sau recuperarea înregistrării în SEAP răspund pentru corectitudinea datelor şi informaţiilor transmise în cadrul procedurii de înregistrare şi/sau reînnoire a înregistrării şi au obligaţia de a transmite operatorului SEAP orice modificare survenită în legătură cu aceste date şi informaţii, în termen de cel mult 3 zile lucrătoare de la producerea respectivelor modificări.""",
    #       """a) întreprinde demersurile necesare pentru înregistrarea/reînnoirea/recuperarea înregistrării autorităţii contractante în SEAP sau recuperarea certificatului digital, dacă este cazul;""",
    #       """(1) Au calitatea de autoritate contractantă în sensul prezentei legi:""",
    #       """a) autorităţile şi instituţiile publice centrale sau locale, precum şi structurile din componenţa acestora care au delegată calitatea de ordonator de credite şi care au stabilite competenţe în domeniul achiziţiilor publice;""",
    #       """b) organismele de drept public;""",
    #       """c) asocierile formate de una sau mai multe autorităţi contractante dintre cele prevăzute la lit. a) sau b).""",
    #       """1. "autorităţi contractante" înseamnă statul, autorităţile regionale sau locale, organismele de drept public sau asociaţiile formate din una sau mai multe astfel de autorităţi sau din unul sau mai multe astfel de organisme de drept public;""",
    #       """2. "autorităţi guvernamentale centrale" înseamnă autorităţile contractante enumerate în anexa I şi, în măsura în care sunt operate corecţii sau modificări la nivel naţional, entităţile care le succed acestora;""",
    #       """3. "autorităţi contractante regionale sau locale" înseamnă toate autorităţile contractante care nu sunt autorităţi guvernamentale centrale;""",
    #       """10. "operator economic" înseamnă orice persoană fizică sau juridică sau o entitate publică sau grup de astfel de persoane şi/sau entităţi, inclusiv orice asociere temporară de întreprinderi, care oferă execuţia de lucrări şi/sau o lucrare, furnizarea de produse sau prestarea de servicii pe piaţă;""",
    #       """11. "ofertant" înseamnă un operator economic care a depus o ofertă;""",

    #       ],
    #   'DEBUG' : True,
    # },
    
    # {
    #   'QUERY' : 'Există posibilitatea înlocuirii unui membru al unei asocieri temporare de operatori economici, căreia i-a fost atribuit un contract/acord-cadru, cu un alt operator economic care îndeplineşte criteriile de calificare şi selecţie stabilite iniţial, ca urmare a unei succesiuni cu titlu universal în cadrul unui proces de divizare?',
    #   'TOP_N' : 0,
    #   'DOCUMENTS': [
    #       """(2) Contractele subsecvente încheiate după intrarea în vigoare a prezentei ordonanţe de urgenţă, pe perioada de derulare a acordului-cadru, se supun legii în vigoare de la data încheierii acestora. """,
    #       """(1) Contractele de achiziţie publică/Acordurile- cadru pot fi modificate, fără organizarea unei noi proceduri de atribuire, în următoarele situaţii:""",
    #       """d) atunci când contractantul cu care autoritatea contractantă a încheiat iniţial contractul de achiziţie publică este înlocuit de un nou contractant, în una dintre următoarele situaţii:""",
    #       """(i) ca urmare a unei clauze de revizuire sau a unei opţiuni stabilite de autoritatea contractantă potrivit lit. a) şi alin. (2);""",
    #       """(ii) drepturile şi obligaţiile contractantului iniţial rezultate din contractul de achiziţie publică sunt preluate, ca urmare a unei succesiuni universale sau cu titlu universal în cadrul unui proces de reorganizare, inclusiv prin fuziune, divizare, achiziţie sau insolvenţă, de către un alt operator economic care îndeplineşte criteriile de calificare şi selecţie stabilite iniţial, cu condiţia ca această modificare să nu presupună alte modificări substanţiale ale contractului de achiziţie publică şi să nu se realizeze cu scopul de a eluda aplicarea procedurilor de atribuire prevăzute de prezenta lege;""",
    #       """(iii) în cazul în care autoritatea contractantă îşi asumă obligaţiile contractantului principal faţă de subcontractanţii acestuia, respectiv aceştia faţă de autoritatea contractantă;""",
    #       """jj) operator economic - orice persoană fizică sau juridică, de drept public ori de drept privat, sau grup ori asociere de astfel de persoane, inclusiv orice asociere temporară formată între două ori mai multe dintre aceste entităţi, care oferă în mod licit pe piaţă executarea de lucrări, furnizarea de produse ori prestarea de servicii, şi care este/sunt stabilită/stabilite în: """
    #       ],
    #   'DEBUG' : True,
    # },
  ]
  
  import pandas as pd
  
  xls = pd.ExcelFile("C:\Proiecte\LegeAI\Date\Task8\get_mark-2.xlsx")
  num_tests = 10
  for i in range(1, num_tests):
        
         sheet = 't' + str(i+1)    
         df = pd.read_excel(xls, sheet)
         query = df.columns[1]
         df.columns = ['idx', 'text']
        
         documents = df.query('idx.str.contains("doc", na=False) and text != "…"', engine='python').text
         documents = list(documents)#[:3]
        
         test_dict = {
             'QUERY' : query,
             'TOP_N' : 0,
             'DOCUMENTS' : documents,
             'DEBUG' : True
             }
        
         inputs_to_test.append(test_dict)
         break
        
   # for i, inp in enumerate(inputs_to_test):
      
   #     j = i + 1
   #     while len(inp['DOCUMENTS']) < 6:
   #         if j == len(inputs_to_test):
   #             j = 0
          
   #         inp['DOCUMENTS'].append(inputs_to_test[j]['DOCUMENTS'][0])
   #         j = j + 1

  for i,_input in enumerate(inputs_to_test):
      result = w.execute(inputs=_input, counter=i)
      print(result)
