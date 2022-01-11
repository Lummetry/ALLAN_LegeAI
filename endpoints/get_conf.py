# -*- coding: utf-8 -*-

from libraries.model_server_v2 import FlaskWorker

import constants as ct
import numpy as np
import spacy
import re


_CONFIG = {
  'SPACY_MODEL' : 'ro_core_news_md',
  'DEBUG' : True
 }


CNP_REG1 = "[0-9]{13}"
CNP_REG2 = "[1-8][0-9]{2}[0-1][0-9][0-3][0-9]{7}"

CNP_NO_CHECK = 1
CNP_BASIC_CHECK = 2
CNP_FULL_CHECK = 3

MATCHED_NERS = {'PERSON', 'LOC'}

class GetConfWorker(FlaskWorker):
    """
    Implementation of the worker for GET_CONFIDENTIAL endpoint
    """
    
    
    def __init__(self, **kwargs):
      super(GetConfWorker, self).__init__(**kwargs)
      return

    def _load_model(self):
    
        # Load Romanian spaCy dataset
        try:
            self.nlp_model = spacy.load(self.config_worker['SPACY_MODEL'])
        except OSError:
            spacy.cli.download(self.config_worker['SPACY_MODEL'])
            self.nlp_model = spacy.load(self.config_worker['SPACY_MODEL'])   
        self._create_notification('LOAD', 'Loaded spaCy model')
            
            
        return
    
    def check_cnp(self, cnp):
        """ Check for a well formed CNP """
        
        cnp_const = '279146358279'
        check_digit = int(cnp[-1])
        
        s = 0
        for i, d in enumerate(cnp[:-1]):
            cnp_digit = int(d)
            const_digit = int(cnp_const[i])
            s += cnp_digit * const_digit
            
        r = s % 11
        
        if ((r < 10 and r == check_digit) or 
            (r == 10 and check_digit == 1)):
            return True
        
        return False
    
    
    def match_cnp(self, text, 
                  check_rigidity=CNP_FULL_CHECK
                ):
        """ Return the position of all the matches for CNP in a text. """
        
        # Find all matches
        if check_rigidity == CNP_NO_CHECK:
            cnp_reg = CNP_REG1
        else:
            cnp_reg = CNP_REG2
        
        matches = re.findall(cnp_reg, text)
        
        res = {}
        for match in matches:
            if check_rigidity < CNP_FULL_CHECK or (check_rigidity == CNP_FULL_CHECK and self.check_cnp(match)):
                # res[text.find(match)] = 'CNP'
                res[text.find(match)] = match
                
        return res
    
    def match_ner(self, nlp, text,
                  ners=MATCHED_NERS
                 ):
        """ Return the position of spaCy named entities in a text. """
        matches = {}    
        
        doc = nlp(text)
        
        for ent in doc.ents:
            if ent.label_ in ners:
                # matches[ent.start_char] = ent.label_
                matches[ent.start_char] = ent.text
                
        return matches   
    
    def _pre_process(self, inputs):
                
        doc = inputs['DOCUMENT']
        if len(doc) < ct.MODELS.TAG_MIN_INPUT:
          raise ValueError("Document: '{}' is below the minimum of {} words".format(
            doc, ct.MODELS.TAG_MIN_INPUT))
    
        return doc

    def _predict(self, prep_inputs):
        
        doc = prep_inputs    
        
        matches = {}
        
        # Match CNPS
        matches.update(self.match_cnp(doc))
        
        # Match NERs
        matches.update(self.match_ner(self.nlp_model, doc))    
              
        return matches

    def _post_process(self, pred):
        
        matches = pred
        idxs = list(matches.keys())
        
        res = {}
        res['results'] = matches
        
        return res


if __name__ == '__main__':
  from libraries import Logger

  l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
  eng = GetConfWorker(log=l, default_config=_CONFIG, verbosity_level=1)
  
  test = {
        'DOCUMENT': """Un contribuabil al cărui cod numeric personal este 2548016600768 va completa caseta "Cod fiscal" astfel:""",
      
        # 'DOCUMENT': """Se desemnează domnul Cocea Radu, avocat, domiciliat în municipiul Bucureşti, Bd. Laminorului nr. 84, sectorul 1, legitimat cu C.I. seria RD nr. 040958, eliberată la data de 16 septembrie 1998 de Secţia 5 Poliţie Bucureşti, CNP 1561119034963, în calitate de administrator special.""", 
        
        # 'DOCUMENT': """Cod numeric personal: 1505952103022. Doi copii minori înregistraţi în documentul de identitate.""",
        
        # 'DOCUMENT': """Bătrîn Cantemhir-Marian, porcine, Str. Cardos Iacob nr. 34, Arad, judeţul Arad, 1850810020101.
        # Almăjanu Steliana, porcine, Comuna Peretu, judeţul Teleorman, 2580925341708.""",
                
#         'DOCUMENT': """Poziţia comună nr. 150/2007 de reînnoire a măsurilor stabilite în sprijinul punerii efective în aplicare a mandatului Tribunalului Penal Internaţional pentru fosta Iugoslavie (TPII)
# POZIŢIA COMUNĂ 2007/150/PESC A CONSILIULUI<BR/>din 5 martie 2007<BR/>de reînnoire a măsurilor stabilite în sprijinul punerii efective în<BR/>aplicare a mandatului Tribunalului Penal Internaţional pentru fosta<BR/>Iugoslavie (TPII)
# CONSILIUL UNIUNII EUROPENE,
# Având în vedere Tratatul privind Uniunea Europeană, în special articolul 15,
# întrucât:
# La 30 martie 2004, Consiliul a adoptat Poziţia comună 2004/293/PESC de reînnoire a măsurilor stabilite în sprijinul punerii efective în aplicare a mandatului Tribunalului Penal Internaţional pentru fosta Iugoslavie (TPII) (<sup>1</sup>). Aceste măsuri au fost reînnoite prin Poziţia comună 2006/204/PESC (<sup>2</sup>) şi urmează să expire la 16 martie 2007.
# Domnul Ratomir SPAJIC, decedat, ar trebui înlăturat de pe lista inclusă în anexa la Poziţia comună 2004/293/PESC.
# Consiliul consideră că este necesar să reînnoiască măsurile impuse prin Poziţia comună 2004/293/PESC pentru o perioadă suplimentară de 12 luni,
# ADOPTĂ PREZENTA POZIŢIE COMUNĂ:
# ___________
# JO L 94, 31.3.2004, p. 65. Poziţie comună, astfel cum a fost modificată ultima dată prin Decizia 2005/83/PESC (JO L 29, 2.2.2005, p. 50).
# JO L 72, 11.3.2006, p. 15.

# Durata de aplicare a Poziţiei comune 2004/293/PESC se prelungeşte până la 16 martie 2008.

# Lista persoanelor din anexa la Poziţia comună 2004/293/PESC se înlocuieşte cu lista din anexa la prezenta poziţie comună.

# Prezenta poziţie comună produce efecte de la data adoptării.

# Prezenta poziţie comună se publică în Jurnalul Oficial al Uniunii Europene.
# Adoptată la Bruxelles, 5 martie 2007.
# Pentru Consiliu<BR/>Preşedintele<BR/>F. -W. STEINMEIER

# BAGIC, Zeljko
# Fiul lui Josip
# Data şi locul naşterii: 29.3.1960, Zagreb, Croaţia
# Paşaport nr.:
# Carte de identitate nr.:
# Cod numeric personal:
# Alias: Cicko
# Adresă:
# BILBIJA, Milorad
# Fiul lui Svetko Bilbija
# Data şi locul naşterii: 13.8.1956, Sanski Most, Bosnia şi Herţegovina
# Paşaport nr.: 3715730
# Carte de identitate nr.: 03GCD9986
# Cod numeric personal: 1308956163305
# Alias:
# Adresă: Brace Pantica 7, Banja Luka, Bosnia şi Herţegovina
# BJELICA, Milovan
# Data şi locul naşterii: 19.10.1958, Rogatica, Bosnia şi Herţegovina
# Paşaport nr.: 0000148, eliberat la 26.7.1998 la Srpsko Sarajevo (anulat)
# Carte de identitate nr.: 03ETA0150
# Cod numeric personal: 1910958130007
# Alias: Cicko
# Adresă: Societatea CENTREK, Pale, Bosnia şi Herţegovina
# CESIC, Ljubo
# Fiul lui Jozo
# Data şi locul naşterii: 20.2.1958 sau 9.6.1966 (document de referinţă de la Ministerul de Justiţie al Croaţiei), Batin, Posusje, Bosnia şi Herţegovina
# Paşaport nr.:
# Carte de identitate nr.:
# Cod numeric personal:
# Alias: Rojs
# Adresă: V Poljanice 26, Dubrava, Zagreb; alt domiciliu: Novacka 62c, Zagreb, Croaţia
# DILBER, Zeljko
# Fiul lui Drago
# Data şi locul naşterii: 2.2.1955, Travnik, Bosnia şi Herţegovina
# Paşaport nr.:
# Carte de identitate nr.: 185581
# Cod numeric personal:
# Alias:
# Adresă: 17 Stanka Vraza, Zadar, Croaţia
# ECIM, Ljuban
# Data şi locul naşterii: 6.1.1964, Sviljanac, Bosnia şi Herţegovina
# Paşaport nr.: 0144290, eliberat la 21.11.1998 la Banja Luka (anulat)
# Carte de identitate nr.: 03GCE3530
# Cod numeric personal: 0601964100083
# Alias:
# Adresă: Ulica Stevana Mokranjca 26, Banja Luka, Bosnia şi Herţegovina
# JOVICIC, Predrag
# Fiul lui Desmir Jovicic
# Data şi locul naşterii: 1.3.1963, Pale, Bosnia şi Herţegovina
# Paşaport nr.: 4363551
# Carte de identitate nr.: 03DYA0852
# Cod numeric personal: 0103963173133
# Alias:
# Adresă: Milana Simovica 23, Pale, Bosnia şi Herţegovina
# KARADZIC, Aleksandar
# Data şi locul naşterii: 14.5.1973, Sarajevo Centar, Bosnia şi Herţegovina
# Paşaport nr.: 0036395 (expirat la 12.10.1998)
# Carte de identitate nr.:
# Cod numeric personal:
# Alias: Sasa
# Adresă:
# KARADZIC, Ljiljana (numele anterior căsătoriei: ZELEN)
# Fiica lui Vojo şi Anka
# Data şi locul naşterii: 27.11.1945, Sarajevo Centar, Bosnia şi Herţegovina
# Paşaport nr.:
# Carte de identitate nr.:
# Cod numeric personal:
# Alias:
# Adresă:
# KESEROVIC, Dragomir
# Fiul lui Slavko
# Data şi locul naşterii: 8.6.1957, Piskavica/Banja Luka, Bosnia şi Herţegovina
# Paşaport nr.: 4191306
# Carte de identitate nr.: 04GCH5156
# Cod numeric personal: 0806957100028
# Alias:
# Adresă:
# KIJAC, Dragan
# Data şi locul naşterii: 6.10.1955, Sarajevo, Bosnia şi Herţegovina
# Paşaport nr.:
# Carte de identitate nr.:
# Cod numeric personal:
# Alias:
# Adresă:
# KOJIC, Radomir
# Fiul lui Milanko şi Zlatana
# Data şi locul naşterii: 23.11.1950, Bijela Voda, cantonul Sokolac, Bosnia şi Herţegovina
# Paşaport nr.: 4742002, eliberat în 2002 la Sarajevo (expiră în 2007)
# Carte de identitate nr.: 03DYA1935. Eliberată la 7.7.2003 la Sarajevo
# Cod numeric personal: 2311950173133
# Alias: Mineur sau Ratko
# Adresă: 115 Trifka Grabeza, Pale, sau Hotel KRISTAL, Jahorina, Bosnia şi Herţegovina
# KOVAC, Tomislav
# Fiul lui Vaso
# Data şi locul naşterii: 4.12.1959, Sarajevo, Bosnia şi Herţegovina
# Paşaport nr.:
# Carte de identitate nr.:
# Cod numeric personal: 0412959171315
# Alias: Tomo
# Adresă: Bijela, Muntenegru şi Pale, Bosnia şi Herţegovina
# KRASIC, Petar
# Data şi locul naşterii:
# Paşaport nr.:
# Carte de identitate nr.:
# Cod numeric personal:
# Alias:
# Adresă:
# KUJUNDZIC, Predrag
# Fiul lui Vasilija
# Data şi locul naşterii: 30.1.1961, Suho Pole, Doboj, Bosnia şi Herţegovina
# Paşaport nr.:
# Carte de identitate nr.: 03GFB1318
# Cod numeric personal: 3001961120044
# Alias: Predo
# Adresă: Doboj, Bosnia şi Herţegovina
# LUKOVIC, Milorad Ulemek
# Data şi locul naşterii: 15.5.1968, Belgrad, Serbia
# Paşaport nr.:
# Carte de identitate nr.:
# Cod numeric personal:
# Alias: Legija (identitate falsă IVANIC, Zeljko)
# Adresă: se sustrage urmăririi
# MAKSAN, Ante
# Fiul lui Blaz
# Data şi locul naşterii: 7.2.1967, Pakostane, lângă Zadar, Croaţia
# Paşaport nr.: 1944207
# Carte de identitate nr.:
# Cod numeric personal:
# Alias: Djoni
# Adresă: Proloska 15, Pakostane, Zadar, Croaţia
# MALIS, Milomir
# Fiul lui Dejan Malis
# Data şi locul naşterii: 3.8.1966, Bjelice
# Paşaport nr.:
# Carte de identitate nr.:
# Cod numeric personal: 0308966131572
# Alias:
# Adresă: Vojvode Putnika, Foca/Srbinje, Bosnia şi Herţegovina
# MANDIC, Momcilo
# Data şi locul naşterii: 1.5.1954, Kalinovik, Bosnia şi Herţegovina
# Paşaport nr.: 0121391, eliberat la 12.5.1999 la Srpsko Sarajevo, Bosnia şi Herţegovina (anulat)
# Carte de identitate nr.:
# Cod numeric personal: 0105954171511
# Alias: Momo
# Adresă: Discoteca GITROS din Pale, Bosnia şi Herţegovina
# MARIC, Milorad
# Fiul lui Vinko Maric
# Data şi locul naşterii: 9.9.1957, Visoko, Bosnia şi Herţegovina
# Paşaport nr.: 4587936
# Carte de identitate nr.: 04GKB5268
# Cod numeric personal: 0909957171778
# Alias:
# Adresă: Vuka Karadzica 148, Zvornik, Bosnia şi Herţegovina
# MICEVIC, Jelenko
# Fiul lui Luka şi Desanka, numele anterior căsătoriei: SIMIC
# Data şi locul naşterii: 8.8.1947, Borci lângă Konjic, Bosnia şi Herţegovina
# Paşaport nr.: 4166874
# Carte de identitate nr.: 03BIA3452
# Cod numeric personal: 0808947710266
# Alias: Filaret
# Adresă: Mănăstirea Milesevo, Serbia
# NINKOVICK, Milan
# Fiul lui Simo
# Data şi locul naşterii: 15.6.1943, Doboj, Bosnia şi Herţegovina
# Paşaport nr.: 3944452
# Carte de identitate nr.: 04GFE3783
# Cod numeric personal: 1506943120018
# Alias:
# Adresă:
# OSTOJIC, Velibor
# Fiul lui Jozo
# Data şi locul naşterii: 8.8.1945, Celebici, Foca, Bosnia şi Herţegovina
# Paşaport nr.:
# Carte de identitate nr.:
# Cod numeric personal:
# Alias:
# Adresă:
# OSTOJIC, Zoran
# Fiul lui Mico Ostojic
# Data şi locul naşterii: 29.3.1961, Sarajevo, Bosnia şi Herţegovina
# Paşaport nr.:
# Carte de identitate nr.: 04BSF6085
# Cod numeric personal: 2903961172656
# Alias:
# Adresă: Malta 25, Sarajevo, Bosnia şi Herţegovina
# PAVLOVIC, Petko
# Fiul lui Milovan Pavlovic
# Data şi locul naşterii: 6.6.1957, Ratkovici, Bosnia şi Herţegovina
# Paşaport nr.: 4588517
# Carte de identitate nr.: 03GKA9274
# Cod numeric personal: 0606957183137
# Alias:
# Adresă: Vuka Karadjica 148, Zvornik, Bosnia şi Herţegovina
# PETRAC, Hrvoje
# Data şi locul naşterii: 25.8.1955, Slavonski Brod, Croaţia
# Paşaport nr.: paşaport croat nr. 01190016
# Carte de identitate nr.:
# Cod numeric personal:
# Alias:
# Adresă:
# POPOVIC, Cedomir
# Fiul lui Radomir Popovic
# Data şi locul naşterii: 24.3.1950, Petrovici
# Paşaport nr.:
# Carte de identitate nr.: 04FAA3580
# Cod numeric personal: 2403950151018
# Alias:
# Adresă: Crnogorska 36, Bileca, Bosnia şi Herţegovina
# PUHALO, Branislav
# Fiul lui Djuro
# Data şi locul naşterii: 30.8.1963, Foca, Bosnia şi Herţegovina
# Paşaport nr.:
# Carte de identitate nr.:
# Cod numeric personal: 3008963171929
# Alias:
# Adresă:
# RADOVIC, Nade
# Fiul lui Milorad Radovic
# Data şi locul naşterii: 26.1.1951, Foca, Bosnia şi Herţegovina
# Paşaport nr.: paşaport vechi nr. 0123256 (anulat)
# Carte de identitate nr.: 03GJA2918
# Cod numeric personal: 2601951131548
# Alias:
# Adresă: Stepe Stepanovica 12, Foca/Srbinje, Bosnia şi Herţegovina
# RATIC, Branko
# Data şi locul naşterii: 26.11.1957, MIHALJEVCI SLAVONSKA POZEGA, Bosnia şi Herţegovina
# Paşaport nr.: 0442022, eliberat la 17.9.1999 la Banja Luka
# Carte de identitate nr.: 03GCA8959
# Cod numeric personal: 2611957173132
# Alias:
# Adresă: Ulica Krfska 42, Banja Luka, Bosnia şi Herţegovina
# ROGULJIC, Slavko
# Data şi locul naşterii: 15.5.1952, SRPSKA CRNJA HETIN, Serbia
# Paşaport nr.: paşaport valabil nr. 3747158, eliberat la 12.4.2002 la Banja Luka. Data expirării: 12.4.2007. Paşaport expirat nr. 0020222 eliberat la 25.8.1988 la Banja Luka. Data expirării: 25.8.2003
# Carte de identitate nr.: 04EFA1053
# Cod numeric personal: 1505952103022
# Alias:
# Adresă: 21 Vojvode Misica, Laktasi, Bosnia şi Herţegovina
# SAROVIC, Mirko
# Data şi locul naşterii: 16.9.1956, Rusanovici-Rogatica, Bosnia şi Herţegovina
# Paşaport nr.: 4363471 eliberat la Srpsko Sarajevo, expiră la 8 octombrie 2008
# Carte de identitate nr.: 04PEA4585
# Cod numeric personal: 1609956172657
# Alias:
# Adresă: Bjelopoljska 42, 71216 Srpsko Sarajevo, Bosnia şi Herţegovina
# SKOCAJIC, Mrksa
# Fiul lui Dejan Skocajic
# Data şi locul naşterii: 5.8.1953, Blagaj, Bosnia şi Herţegovina
# Paşaport nr.: 3681597
# Carte de identitate nr.: 04GDB9950
# Cod numeric personal: 0508953150038
# Alias:
# Adresă: Brigada Trebinjskih, Trebinje, Bosnia şi Herţegovina
# VRACAR, Milenko
# Data şi locul naşterii: 15.5.1956, Nisavici, Prijedor, Bosnia şi Herţegovina
# Paşaport nr.: paşaport valabil nr. 3865548, eliberat la 29.8.2002 la Banja Luka. Data expirării: 29.8.2007. Paşapoarte expirate: 0280280 eliberat la 4.12.1999 la Banja Luka (data expirării: 4.12.2004) şi 0062130 eliberat la 16.9.1998 la Banja Luka, Bosnia şi Herţegovina
# Carte de identitate nr.: 03GCE6934
# Cod numeric personal: 1505956160012
# Alias:
# Adresă: 14 Save Ljuboje, Banja Luka, Bosnia şi Herţegovina
# ZOGOVIC, Milan
# Fiul lui Jovan
# Data şi locul naşterii: 7.10.1939, Dobrusa
# Paşaport nr.:
# Carte de identitate nr.:
# Cod numeric personal:
# Alias:
# Adresă:"""
        
      }
  
  res = eng.execute(inputs=test, counter=1)
  print(res)
