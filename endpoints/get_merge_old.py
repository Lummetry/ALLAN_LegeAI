# -*- coding: utf-8 -*-

from libraries.model_server_v2 import FlaskWorker
import regex as re
import spacy

_CONFIG = {
  'SPACY_MODEL' : 'ro_core_news_lg',
 }


MIN_PASIV_WORDS = 2
MIN_ACTIV_WORDS = 5

STRIP_CHARS = " .,:;\"\'()[]{}"

REMOVE_KEYS = ['elimină', 'eliminat', 'eliminăm']

REPLACE_KEYS = ['înlocuieşte', 'înlocuiesc', 'înlocuim']

ADD_COMMA_KEYS = ['se adaugă virgulă', 'se adauga virgula', 'adaugăm virgulă', 'adaugam virgula']

BEFORE_WORD = -1
AFTER_WORD = 1

DEVINE_KEYS = ['devine', 'va deveni']

COMMON_REPLACE_WORDS = ['denumirea', 'denumirile',
                        'termenul', 'termenii',
                        'pozitia', 'pozitiile', 'poziția', 'pozițiile',
                        'formularea', 'formularile', 'formulările',
                        'cuvântul', 'cuvantul', 'cuvintele'
                       ]

KEYWORDS = REMOVE_KEYS + REPLACE_KEYS + ADD_COMMA_KEYS + DEVINE_KEYS

DELTA_QUOTES = 6

LINK_PATTERN = "~id_link=[^;]*;([^~]*)~"


__VER__='0.3.0.0'
class GetMergeWorker(FlaskWorker):
    """
    Implementation of the worker for GET_MERGE endpoint
    """
    
    
    def __init__(self, **kwargs):
        super(GetMergeWorker, self).__init__(**kwargs)
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
    
    
    def remove_links(self, text):
        """ Remove links from text. """
        
        match = re.search(LINK_PATTERN, text)
        
        while match:
            linkText = match.groups()[0]
            start, end = match.span()
    
            text = text[:start] + linkText + text[end:]
    
            match = re.search(LINK_PATTERN, text)
            
        return text
    
    
    def match_keywords(self, activ, keywords):
        """ Search for keywords in the text. """
        
        match_dict = {}
        
        for key in keywords:
            pos = activ.find(key)
            
            if pos > -1:
                match_dict[pos] = key
                
        return match_dict
    
    def get_common_subsequences_lemmas(self, nlp_text1, nlp_text2,
                                       min_len = 2, top=3
                                      ):
        """ Get the list of common contiguous subsequences between two texts"""
        
        seqs = []
        
        i = 0
        
        while i < len(nlp_text1):
            
            j = 0
            max_len = 0
            while j < len(nlp_text2):
                            
                k = 0
                while i + k < len(nlp_text1) and j + k < len(nlp_text2) and nlp_text1[i + k].lemma_ == nlp_text2[j + k].lemma_:
                    k += 1
                
                
                if k > 0:
                    # If a common sequence was found     
                    seq_text = nlp_text2[j : j+k].text
                    seq_len = len(seq_text)
                    
                    if seq_len > min_len and seq_len > max_len:
                        # If it is the max match for the current starting point in the first text
                        max_len = seq_len
                        max_seq = (nlp_text1[i].idx, nlp_text1[i].idx + len(nlp_text1[i : i+k].text), 
                                   nlp_text2[j].idx, nlp_text2[j].idx + seq_len, 
                                   seq_text)
                    
                j += 1
                
            if max_len > 0:
                
                start_i2, end_i2, start_j2, end_j2, _ = max_seq
                included = False
                
                for (start_i1, end_i1, start_j1, end_j1, _) in seqs:
                    if (start_i1 <= start_i2 and end_i2 <= end_i1) or (start_j1 <= start_j2 and end_j2 <= end_j1):
                        included = True
                        break
                        
                if not included:
                    seqs.append(max_seq)
                
            i += 1
        
        seqs.sort(key=lambda t: t[1] - t[0], reverse=True)
        
        return seqs[:top]
    
    def search_removals(self, pasiv, activ,
                        start_seq, end_seq,
                        matches,
                        check_quotes=True                
                    ):
        """ Search for removals. """
        
        pasiv = pasiv.lower()
        activ = activ.lower()
        
        
        removals = []
        
        # Look for long matches between Pasiv and Activ which appear in the given interval in Activ
        for (start_pasiv, end_pasiv, start_activ, end_activ, match) in matches:
            
                
            current_start = start_seq
            while match in activ[current_start:end_seq]:            
                remove_match = True
                
                if check_quotes:
                    # Check if the match is surrounded in quotes
                    start_quote = False
                    for p in range(max(0, start_activ - DELTA_QUOTES), start_activ):
                        if activ[p] in '\'"':
                            start_quote = True
                            break
                    end_quote = False
                    for p in range(end_activ, min(len(activ), end_activ + DELTA_QUOTES)):
                        if activ[p] in '\'"':
                            end_quote = True
                            break
                            
                    if not start_quote or not end_quote:
                        remove_match = False
                    
                if remove_match:
                               
                    # Eliminate trailing unwanted characters
                    limit1 = start_pasiv     
                    limit2 = end_pasiv
                    while limit2 < len(pasiv) and pasiv[limit2] in STRIP_CHARS:
                        limit2 += 1
                        
                    removals.append(('remove', limit1, limit2))
                    
                # Check rest of the sequence
                current_start = current_start + activ[current_start:end_seq].find(match) + len(match)
        
        return removals
    
    def clean_seq_to_replace(self, seq, activ_nlp):
        """ Clean the sequence to replace. """
        
        start, end = seq
        
        # Remove surrounding punctuation
        while activ_nlp[start].is_punct:
            start += 1
        while activ_nlp[end - 1].is_punct:
            end -= 1
        
        # Remove 'cu' from start
        if activ_nlp[start].text == 'cu':
            start += 1
        
        return start, end
    
    def search_replaces(self, pasiv, activ, activ_nlp,
                        start_char, end_char,
                        matches,
                        check_quotes=True, remove_common=True, find_cu=False
                    ):
        """ Search for replaces. """
        
        new_seqs = []
        replaces = []
        
        # Get the token span corresponding to the character span
        start_tok, end_tok = self.char_to_token_idx(activ_nlp, start_char, end_char)
        
        start_seq = -1
        if find_cu:
            # Find sequences after keyword "cu"    
    
            for t in activ_nlp[start_tok: end_tok]:
                if t.text == "cu":
                    if start_seq > -1:
                        end_seq = t.i
                        new_seqs.append((start_seq, end_seq))
                    start_seq = t.i + 1 
    
        if start_seq == -1:
            # If no "cu" keyword was found
            start_seq = start_tok
                
        # Append last sequence
        new_seqs.append((start_seq, end_tok))
             
        if check_quotes:
            # Isolate only sequences containing quotes
            
            quote_seqs = []
            
            for i, seq in enumerate(new_seqs):
                
                new_start, new_end = None, None
                j = seq[0]
                while j < seq[1]:
                    token = activ_nlp[j]
                    
                    # Find first quote
                    if token.is_quote:
                        new_start = token.i + 1
                        
                        k = new_start+1
                        while k < seq[1]:
                            token2 = activ_nlp[k]
                            
                            # Find second quote
                            if token2.is_quote:
                                new_end = token2.i
                                quote_seqs.append((new_start, new_end))
                                new_start = None
                                break
                            else:
                                k += 1
                              
                        j = k + 1
                    else:
                        j += 1
                        
                if new_start and new_end:
                    # If a quote was found
                    new_seqs[i] = (new_start, new_end)                    
            
            
            
            if len(quote_seqs) == 0:
                # If no quote sequences were found, stick to the initial ones
                pass
            else:
                new_seqs = quote_seqs   
                    
        if remove_common:
            # Remove some common words from the sequences
           
            for i, seq in enumerate(new_seqs):
                
                new_start = None
                for token in activ_nlp[seq[0] : seq[1]]:
                    # Find first quote
                    if token.text in COMMON_REPLACE_WORDS:
                        new_start = token.i + 1
                        break
                        
                if new_start:
                    # If a quote was found
                    new_seqs[i] = (new_start, seq[1])   
                    
                    
        # Remove sequences that are the same as the match
        i = 0
        while i < len(new_seqs):
            is_match = False
            
            seq_text = activ_nlp[new_seqs[i][0] : new_seqs[i][1]].text.lower()
            for m in matches:
                if seq_text == m[4]:
                    # If the sequence is the same as a match
                    is_match = True
                    break
                    
            if is_match:
                del new_seqs[i]
            else:
                i += 1
                    
        if new_seqs:
            
            for i in range(len(new_seqs)):
                
                # Clean the phrase
                seq_start, seq_end = self.clean_seq_to_replace(new_seqs[i], activ_nlp)
                                
                # Get the phrase to replace
                start_idx = activ_nlp[seq_start].idx
                end_idx = activ_nlp[seq_end - 1].idx + len(activ_nlp[seq_end - 1].text)          
                to_replace = activ[start_idx:end_idx]
                
                # If there are matches left
                if len(matches) > i:
                    
                    # Replace the initial occurange of phrase to be replaced
                    replaces.append(('replace', matches[i][0], matches[i][1], to_replace))
    
                    # Find all other occurances of phrase to be replaced
                    for m in re.finditer(matches[i][4], pasiv.lower()):
                        # Add replace action
                        if m.start() != matches[i][0] and m.end() != matches[i][1]:
                            replaces.append(('replace', m.start(), m.end(), to_replace))
                     
                # TODO: Only does first replace
                break
                    
        return replaces
    
    def search_add_comma(self, pasiv, activ, activ_nlp,
                         start_char, end_char,
                         
                    ):
        """ Search for comma additions. """
                
        # Get the token span corresponding to the character span
        start_tok, end_tok = self.char_to_token_idx(activ_nlp, start_char, end_char)
        
        # Find position keyword
        comma_position = 0
        for i in range(start_tok, -1, -1):
            tok = activ_nlp[i]
            
            if tok.text in ['după', 'dupa']:
                comma_position = AFTER_WORD
                start_tok_pos = i + 1
                break
                
            elif tok.text in ['înainte', 'inainte', 'înaintea']:
                comma_position = BEFORE_WORD
                start_tok_pos = i + 1
                if activ_nlp[i + 1].text == 'de':
                    start_tok_pos += 1
                break
                
        if comma_position:
            
            # First look for a sequence in quotes
            start_quote, end_quote = None, None
            for i, tok in enumerate(activ_nlp[start_tok_pos : start_tok]):
                if tok.is_quote:
                    if start_quote:
                        end_quote = start_tok_pos + i
                        # End of quote
                        break
                    else:
                        # Start of quote
                        start_quote = start_tok_pos + i + 1
                     
            if start_quote and end_quote:
                # If a quote was found
                quote_seq = activ_nlp[start_quote : end_quote].text
                
                return ('comma', comma_position, comma_position, quote_seq)
            
            else:
                # No quote found, take entire sequence
                start_seq = start_tok_pos + 1
                end_seq = start_tok
                
                # Remove some common words from the start
                if activ_nlp[start_seq].text in COMMON_REPLACE_WORDS:
                    start_seq += 1
                    
                quote_seq = activ_nlp[start_seq : end_seq].text
                
                return ('comma', comma_position, comma_position, quote_seq)       
            
        return None    
    
    def add_comma(self, pasiv, action):
        """ Apply an Add Comma action to the Pasiv text. """
        
        seq = action[3]
        comma_pos = action[1]
        
        new_pasiv = pasiv    
        text = pasiv
        while True:
            pos = text.find(seq)    
            if pos == -1:
                break
                
            if comma_pos == AFTER_WORD:
                new_pasiv = new_pasiv[:pos + len(seq)] + ',' + new_pasiv[pos + len(seq) :]
                
            elif comma_pos == BEFORE_WORD:
                new_pasiv = new_pasiv[:pos - 1] + ',' + new_pasiv[pos-1:]
                
            
            text = pasiv[pos + 1:]
        
        return new_pasiv
    
    def is_date(self, ent):
        """ Check if an entity represents a date. """
        
        if ent.label_ == 'DATETIME':
            digit_count = len([c for c in ent.text if c.isdigit()])
            if digit_count >= 4 and digit_count <= 8:
                return True
            
        return False
    
    def search_devine(self, pasiv, activ, pasiv_nlp, activ_nlp
                    ):
        """ Search for replaces. """
        
        date_ents = []
        money_ents = []
        for ent in activ_nlp.ents:
            if self.is_date(ent):
                date_ents.append(ent)
            elif ent.label_ == 'MONEY':
                money_ents.append(ent)
            
        # Handle only the basic case, just one date or money entity
        activ_ent = None
        if len(date_ents) == 1 and len(money_ents) == 0:
            activ_ent = date_ents[0]
            ent_type = 'DATETIME'
        
        elif len(date_ents) == 0 and len(money_ents) == 1:
            activ_ent = money_ents[0]
            ent_type = 'MONEY'
            
        if activ_ent is None:
            return None
            
        # Search for same entity in Pasiv
        pasiv_ents = []
        for ent in pasiv_nlp.ents:
            if ent_type == ent.label_ and ((ent_type == 'DATETIME' and self.is_date(ent)) or ent_type == 'MONEY'):
                pasiv_ents.append(ent)
                
        # If only one similar entity was found
        if len(pasiv_ents) == 1:
            pasiv_ent = pasiv_ents[0]        
            return ('devine', pasiv_ent.start_char, pasiv_ent.end_char, activ_ent.text) 
        
        return None
    
    def char_to_token_idx(self, doc, start_char, end_char):
        """ Find the token indexes corresponding to a sequence expressed in character indexes. """
        
        for token in doc:
            if token.idx >= start_char:
                span_start = token.i
                span_end = span_start + 1            
                
                for token2 in doc[span_start:]:
                    if token2.idx + len(token2.text) >= end_char:                    
                        span_end = token2.i + 1
                        break
                        
                return span_start, span_end
        
        return None, None
    
    def find_keyword_span(self, doc, keyword):
        """ Find the tokens representing a keyword. """
        
        start_idx = doc.text.find(keyword)
        
        if start_idx == -1:
            return None, None
        
        end_idx = start_idx + len(keyword)
        
        # Get the token span corresponding to the character span
        span_start, span_end = self.char_to_token_idx(doc, start_idx, end_idx)
        
        return span_start, span_end
    
    def check_subject(self, activ_nlp, matches, keyword):
        """Check if matches represent subject phrases. """
        
        # Get position of the keyword in the Doc
        start_idx, end_idx = self.find_keyword_span(activ_nlp, keyword)
        
        if start_idx is None:
            return None
        
        # Get the position of the Subject subtree
        start_subtree, end_subtree = None, None
        for token in activ_nlp[start_idx : end_idx]:
            if token.pos_ == 'VERB' or token.pos_ == 'AUX':
                for t in token.children:
                    if 'nsubj' in t.dep_:
                        start_subtree = next(t.subtree).idx
                        for end_subtree in t.subtree:
                            pass
                        end_subtree = end_subtree.idx + len(end_subtree.text)
                        
        if start_subtree is None:
            return None
                        
        # Check matches
        subject_matches = []
        
        for match in matches:
            start_match = match[2]
            end_match = match[3]
            
            # If the match intersects the subject subtree
            if min(start_match, start_subtree) < max(end_match, end_subtree):
                subject_matches.append(match)
                
        return subject_matches
    
    def update_actions(self, actions, current_action):
        """ Update the rest of the action positions according to the current action. """
        
        start_current = current_action[1]
        end_current = current_action[2]
        
        i = 0
        while i < len(actions):
            action = actions[i]    
            start = action[1]
            end = action[2]
            
            if current_action[0] == 'remove':
                
                if min(start, start_current) <= max(end, end_current):
                    # If the two actions intersect, drop the second one
                    del actions[i]
                    continue
                            
                # Update the sequence positions if they occur after the removed sequence
                seq_len = end_current - start_current
                if start > end_current:
                    start = start - seq_len
                if end > end_current:
                    end = end - seq_len
                    
                   
                new_action = list(actions[i])
                new_action[1] = start
                new_action[2] = end
                actions[i] = tuple(new_action)
                    
                i += 1
                
            elif current_action[0] == 'replace' or current_action[0] == 'devine':
                            
                # Update the sequence positions if they occur after the removed sequence
                delta_len = len(current_action[3]) - (end_current - start_current)
                if start > end_current:
                    start = start + delta_len
                if end > end_current:
                    end = end + delta_len
                   
                new_action = list(actions[i])
                new_action[1] = start
                new_action[2] = end
                actions[i] = tuple(new_action)
                    
                i += 1
            
            elif current_action[0] == 'comma':
                            
                # Update the sequence positions if they occur after the removed sequence
                if start > end_current:
                    start = start + 1
                if end > end_current:
                    end = end + 1
                   
                new_action = list(actions[i])
                new_action[1] = start
                new_action[2] = end
                actions[i] = tuple(new_action)
                    
                i += 1
                
            else:
                i += 1
                
        return actions
    
    def apply_transformations(self, pasiv, activ,
                              pasiv_nlp, activ_nlp,
                              check_subj=False
                             ):
        """ Apply transformations in Activ on Pasiv. """
        
        # Get longest common subsequences
        matches = self.get_common_subsequences_lemmas(pasiv_nlp, activ_nlp, top=3)
            
        # Get action keywords in Activ
        keywords = self.match_keywords(activ, KEYWORDS)
        key_items = list(keywords.items())
        # Sort ascending according to position
        key_items.sort(key=lambda t: t[0])
        
        if self.debug:
            print(matches, keywords)
            
        # Find how to apply actions
        actions = []
            
        for i, (pos, key) in enumerate(key_items):  
                
    
            # Check that matches are part of the subject for the keyword
            if check_subj:
                matches = self.check_subject(activ_nlp, matches, key)
                
            if matches:
    
                # Get the interval in Activ until the next keyword
                start_seq = pos + 1
                if i < len(key_items) - 1:
                    end_seq = key_items[i + 1][0]
                else:
                    end_seq = len(activ)  
    
                # REMOVE
                if key in REMOVE_KEYS:
                    actions.extend(self.search_removals(pasiv, activ, start_seq, end_seq, matches, 
                                                        check_quotes=True))
                
                # REPLACE
                elif key in REPLACE_KEYS:
                    actions.extend(self.search_replaces(pasiv, activ, activ_nlp, start_seq, end_seq, matches, 
                                                        check_quotes=True, remove_common=True))
                
                # ADD COMMA
                elif key in ADD_COMMA_KEYS:
                    action = self.search_add_comma(pasiv, activ, activ_nlp, start_seq - 1, end_seq)
                    if action:
                        actions.append(action)
                
                # DEVINE
                elif key in DEVINE_KEYS:
                    action = self.search_devine(pasiv, activ, pasiv_nlp, activ_nlp)
                    if action:
                        actions.append(action)  
                    
        if self.debug:
            print(actions)
                    
        # Apply remaining actions
        transf = pasiv
        for i, action in enumerate(actions):
                
            if action[0] == 'remove':
                transf = transf[:action[1]] + transf[action[2]:]
                
            elif action[0] == 'replace':
                transf = transf[:action[1]] + action[3] + transf[action[2]:] 
            
            elif action[0] == 'comma':
                transf = self.add_comma(transf, action)  
            
            elif action[0] == 'devine':
                transf = transf[:action[1]] + action[3] + transf[action[2]:]  
                
            # Update the rest of the actions
            new_actions = self.update_actions(actions[i+1:], action)
            actions[i+1:] = new_actions          
                
        return transf

    
    
    def _pre_process(self, inputs):
                
        pasiv = inputs['PASIV']
        num_words = len(pasiv.split(' '))
        
        if num_words < MIN_PASIV_WORDS:
           raise ValueError("Document Pasiv is below the minimum of {} words".format(MIN_PASIV_WORDS))
                       
        activ = inputs['ACTIV']
        num_words = len(activ.split(' '))
        
        if num_words < MIN_ACTIV_WORDS:
           raise ValueError("Document Activ is below the minimum of {} words".format(MIN_ACTIV_WORDS))
           
        # Process documents
        pasiv = self.remove_links(pasiv).strip()
        pasiv_nlp = self.nlp_model(pasiv.lower())
        activ = self.remove_links(activ).strip()
        activ_nlp = self.nlp_model(activ.lower())
           
        self.debug = bool(inputs.get('DEBUG', False))
    
        return pasiv, activ, pasiv_nlp, activ_nlp

    def _predict(self, prep_inputs):
        
        pasiv, activ, pasiv_nlp, activ_nlp = prep_inputs
        
        # Transform the Pasiv document accordind to the Activ document
        transf = self.apply_transformations(pasiv, activ, pasiv_nlp, activ_nlp)
              
        return transf

    def _post_process(self, pred):
        
        transf = pred
        
        res = {}
        
        res['results'] = transf
        
        return res


if __name__ == '__main__':
    from libraries import Logger
    
    l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
    eng = GetMergeWorker(log=l, default_config=_CONFIG, verbosity_level=1)
  
    test = {
        # 'PASIV' : """stabileşte repertoriul cinematografic al filmelor din producţia naţională şi străine destinate exploatării în reţeaua cinematografică; asigură fondul de copii de filme şi distribuirea lor în reţeaua cinematografică în vederea realizării programelor de activitate ale acesteia, în condiţiile utilizării şi gospodăririi raţionale şi eficiente a mijloacelor economice pe care le are la dispoziţie. Distribuţia filmelor în premieră în Bucureşti se va face concomitent în cinematografele proprii ale regiei şi cele ale Centrului Naţional al Cinematografiei, iar în oraşele Constanţa şi Piteşti, alternativ;""",        
        # 'ACTIV' : """la art. 4 pct. 4.1. se elimină ultima frază: Distribuţia filmelor în premieră în Bucureşti se va face concomitent în cinematografele proprii ale regiei şi cele ale Centrului Naţional al Cinematografiei, iar în oraşele Constanţa şi Piteşti, alternativ;""",
        
        # 'PASIV' : """Carne şi preparate din carne""",
        # 'ACTIV' : """Cu aceeaşi dată se abrogă alin. 2 al art. 1, poziţia 1. "Bovine (tineret şi adulte)" din anexa nr. 1, precum şi anexa nr. 2 la Hotărârea Guvernului nr. 197 bis din 30 aprilie 1993, iar poziţia "Carne şi preparate din carne" din anexa la Hotărârea Guvernului nr. 206/1993 se înlocuieşte cu poziţia "Carne de porcine şi de pasăre. Preparate din carne".""",
        
        # 'PASIV' : """în rezervaţiile istorice şi de arhitectură, stabilite potrivit legii, sau în cazul lucrărilor care modifică monumentele de orice natură, solicitantul va obţine avizul Comisiei naţionale pentru protecţia monumentelor, ansamblurilor şi siturilor istorice sau al Departamentului pentru urbanism şi amenajarea teritoriului, în zonele de protecţie ale acestora;""",
        # 'ACTIV' : """La art. 7 lit. a) şi b), art. 27 alin. 3, art. 40 şi în anexa din Legea nr. 50/1991 se înlocuiesc denumirile: "Departamentul pentru urbanism şi amenajarea teritoriului" cu "Ministerul Lucrărilor Publice şi Amenajării Teritoriului"; "Ministerul Mediului" cu "Ministerul Apelor, Pădurilor şi Protecţiei Mediului" şi, respectiv, "Ministerul Comerţului şi Turismului" cu "Ministerul Turismului", iar la art. 38 se elimină "Departamentul pentru urbanism şi amenajarea teritoriului".""",

        # 'PASIV' : """Dacă, în urma admiterii acţiunii, autoritatea administrativă este obligată să înlocuiască sau să modifice actul administrativ, să elibereze un certificat, o adeverinţă sau orice alt înscris, executarea hotărârii definitive se va face în termenul prevăzut în cuprinsul ei, iar în lipsa unui astfel de termen, în cel mult 30 de zile de la data rămînerii definitive a hotărîrii.""",
        # 'ACTIV' : """La articolul 16 alineatul 1, noţiunea "hotărâre definitivă" se înlocuieşte cu "hotărâre irevocabilă".""",
        
        # 'PASIV' : """Pentru organizarea de consfătuiri, analize anuale, întruniri cu caracter metodic, schimburi de experienţă, cursuri de perfecţionare, conferinţe de presă şi alte acţiuni similare, Ministerul Tineretului şi Sportului Departamentul sportului -, Comitetul Olimpic Român, federaţiile sau unităţile sportive, precum şi oficiile judeţene pentru tineret şi sport pot suporta cheltuieli de participare ale invitaţilor (transport, cazare şi scoateri din producţie, după caz), şi cheltuieli de întreţinere la nivelul diurnei legal stabilite, în funcţie de condiţiile în care se organizează activitatea respectivă, precum şi cheltuieli de organizare (chirie sală, amplificare, rechizite, aparatură video, casete video, transport materiale).""",
        # 'ACTIV' : """Formulările "Departamentul sportului" şi "oficiile judeţene pentru tineret şi sport", cuprinse în ~normele~ aprobate prin Hotărârea Guvernului ~id_link=653732;nr. 280/1994~, se înlocuiesc cu "Ministerul Tineretului şi Sportului", respectiv "direcţiile judeţene pentru tineret şi sport".""",
        
        # 'PASIV' : """Împrumutatul va proceda astfel: (i) va înainta Băncii, nu mai tîrziu de 31 decembrie 1992, pentru analiză şi comentariu, un plan de acţiune elaborat pe baza recomandărilor din studiul privind costul şi preţurile energiei la care se face referire în partea B (4), (b), (i) a proiectului, cu scopul, între altele, de a reconsidera structura preţurilor sectorului său energetic; şi (ii) imediat după aceea va aduce la îndeplinire planul respectiv cu atenţia şi eficacitatea cuvenite, luînd în considerare şi comentariile Băncii pe baza acestuia.""",
        # 'ACTIV' : """în cadrul paragrafului (a) se înlocuieşte formularea: "studiul privind costul şi preţurile energiei la care se face referire în partea B (4), (b), (i) a proiectului", cu "un studiu privind costul şi preţurile energiei, bazat pe termeni de referinţă satisfăcători pentru Bancă";""",
       
        # 'PASIV' : """Se autorizează Ministerul Finanţelor să emită garanţii pentru credite "revolving", în valoare totală de 65 miliarde lei, pe care băncile comerciale le vor acorda regiilor autonome de interes local şi agenţilor economici sau serviciilor publice care furnizează energie termică la populaţie, în scopul constituirii stocurilor de păcură şi combustibil lichid uşor, prevăzute în Programul energetic pentru perioada 1 aprilie 1996 - 31 martie 1997, aprobat de Guvern.""",
        # 'ACTIV' : """În cuprinsul articolului 1, sintagma "...în valoare totală de 65 miliarde lei..." se înlocuieşte cu sintagma "...în valoare totală de 200 miliarde lei...".""",
       
        # 'PASIV' : """Pentru creanţele bugetare ce urmează a fi stabilite de către instanţele judecătoreşti, organele prevăzute la art. 24, o dată cu introducerea acţiunii împotriva debitorului, precum şi persoanelor care, potrivit legii, răspund solidar cu acesta, pot cere instanţei să dispună, după caz, înfiinţarea popririi asiguratorii asupra veniturilor, a sechestrului asigurator asupra bunurilor mobile, a ipotecii asiguratoare asupra bunurilor imobile ce aparţin celor chemaţi în judecată, precum şi orice altă măsură de indisponibilizare a veniturilor şi bunurilor urmăribile, prevăzută de lege, pentru asigurarea realizării creanţelor bugetare.""",
        # 'ACTIV' : """la art. 16 alin. 1, dupa cuvântul imobile se adaugă virgulă;""",
      
        # 'PASIV' : """Obligaţiunile de tezaur exprimate în dolari S.U.A. cu dobândă sunt scadente la data de 13 august 1997.""",
        # 'ACTIV' : """La punctul 3 - data scadenţei devine 15 august 1997.""",
      
        'PASIV' : """Veniturile nete lunare prevăzute la alin. (1) şi (2) se majorează cu 5.000 lei pentru fiecare membru din familie care face dovada că lucrează în baza unui contract individual de muncă, a unei convenţii civile sau că realizează venituri din activităţi pe cont propriu.""",
        'ACTIV' : """Suma prevăzută la art. 3 alin. (3), la art. 4 alin. (2) şi la art. 12 alin. (2) din Legea nr. 67/1995 devine 5.300 lei.""",
        
        'DEBUG': True
      }
          
    res = eng.execute(inputs=test, counter=1)
    print(res)