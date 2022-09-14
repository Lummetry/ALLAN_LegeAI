# -*- coding: utf-8 -*-

from libraries.model_server_v2 import FlaskWorker
import regex as re
import spacy
import unidecode
import string
import pandas as pd

_CONFIG = {
  'SPACY_MODEL' : 'ro_core_news_lg',
  # The first model is used when Ensamble is turned off
  'NER_MODELS' : ['..\\allan_data\\MergeNER\\model-best-4735-150-150',
                  '..\\allan_data\\MergeNER\\model-best-2299-150-150-d0.1',
                  '..\\allan_data\\MergeNER\\model-best-931-150-150-d0.1',
                  '..\\allan_data\\MergeNER\\model-best-1000',
                  '..\\allan_data\\MergeNER\\model-best-400',
                  ]
 }


MIN_PASIV_WORDS = 2
MIN_ACTIV_WORDS = 5

# Actions
ACTION_PREPOSITIONS = ['cu', 'pana', 'până', 'pînă', 'in', 'în', 'după', 'dupa', 'la', 'din']
ACTION_UNKNOWN = 0
ACTION_INLOCUIESTE = 1
KEYWORDS_INLOCUIESTE = ['înlocuieşte', 'inlocuieste', 'înlocuiesc', 'inlocuiesc', 'înlocui', 'inlocui'
                       'înlocuită', 'inlocuita', 'înlocuit', 'inlocuit']
ACTION_CITESTE = 2
KEYWORDS_CITESTE = ['citi', 'citesc', 'citeste', 'citeşte']
ACTION_PRELUNGESTE_CU = 3
KEYWORDS_PRELUNGESTE_CU = ['prelungeste cu', 'prelungesc cu', 'prelungeşte cu']
ACTION_PRELUNGESTE_PANA = 4
KEYWORDS_PRELUNGESTE_PANA = ['prelungeste până', 'prelungeste pînă', 'prelungeste pana', 'prelungeste pina',
                             'prelungesc până', 'prelungesc pînă', 'prelungesc pana', 'prelungesc pina',
                             'prelungeşte până', 'prelungeşte pînă', 'prelungeşte pana', 'prelungeşte pina']
ACTION_ELIMINA = 5
KEYWORDS_ELIMINA = ['elimină', 'elimina']
ACTION_DEVINE = 6
KEYWORDS_DEVINE = ['devine']
ACTION_PROROGA_PANA = 7
KEYWORDS_PROROGA_PANA = ['proroga pana', 'proroga pina', 'prorogă până', 'prorogă pînă']
ACTION_PROROGA_CU = 8
KEYWORDS_PROROGA_CU = ['proroga cu', 'prorogă cu']
ACTION_CU = 9
KEYWORD_CU = 'cu'
ACTION_MODIFICA = 10
KEYWORDS_MODIFICA = ['modifica', 'modifică']

SAME_ACTION_GROUPS = [[ACTION_INLOCUIESTE, ACTION_CU, ACTION_CITESTE, ACTION_MODIFICA],
                      [ACTION_PRELUNGESTE_PANA, ACTION_PROROGA_PANA],
                      [ACTION_PRELUNGESTE_CU, ACTION_PROROGA_CU]
                      ]

# Period
PERIOD_ERROR = -1
PERIOD_ZILE = 0
KEYWORDS_ZILE = ['zile', 'zi']
PERIOD_LUNI = 1
KEYWORDS_LUNI = ['luni', 'luna']
PERIOD_ANI = 2
KEYWORDS_ANI = ['ani', 'an']

# Punctuation
DOUBLE_PUNCTUATION = r'\({2}|,{2}|\/{2}|"{2}|:{2}|;{2}|\){2}|\\{2}|(?<!\.)\.{2}(?!\.)'
TOO_MANY_DOTS = r'\.{4,}'
SOLO_PUNCTUATION = r'(\.{3})|[\.,\:;\?\!]'
PAIR_PUNCTUATION = r'\(.+\)|\[.+\]|\{.+\}|\".+\"|\'.+\''
MULTIPLE_SPACES = r' {2,}'

# Clean texts
LINK_PATTERN = "~id_link=[^;]*;([^~]*)~"

# Errors
NO_ERROR = 0
ERROR_NO_ACTION = 1
ERROR_MANY_ACTIONS = 2
ERROR_NUM_OLD_NEW = 3
ERROR_ACTION_UKNOWN = 4

# Heuristics    
MODIFICA_CUPRINS_KEYS = ['se modifică şi va avea următorul cuprins', 'se modifica si va avea urmatorul cuprins',
                         'se modifică şi vor avea următorul cuprins', 'se modifica si vor avea urmatorul cuprins']


__VER__='1.1.0.0'
class GetMergeV2Worker(FlaskWorker):
    """
    Second implementation of the worker for GET_MERGE endpoint.
    Uses Machine Learning NER model for Activ.
    """
    
    
    def __init__(self, **kwargs):
        super(GetMergeV2Worker, self).__init__(**kwargs)
        return

    def _load_model(self):
        
        # Load Romanian spaCy dataset
        try:
            self.nlp_model = spacy.load(self.config_worker['SPACY_MODEL'])
        except OSError:
            spacy.cli.download(self.config_worker['SPACY_MODEL'])
            self.nlp_model = spacy.load(self.config_worker['SPACY_MODEL'])   
        self._create_notification('LOAD', 'Loaded spaCy model')
        
        # Load NER models
        self.activ_ners = []
        for ner_path in self.config_worker['NER_MODELS']:
            self.activ_ners.append(spacy.load(ner_path))
            print('Loaded NER model {}'.format(ner_path))
                
        # The first NER model is used if Ensamble is turned off
        self.activ_ner = self.activ_ners[0]
    
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
    
    
    def clean_punctuation(self, text):
        """ Clean unnecessary punctuation resulting from the match replacements. """
            
        # 1. Spaces around pair punctuation
        re_matches = list(re.finditer(PAIR_PUNCTUATION, text))
        re_matches.reverse()
        for re_match in re_matches:
            start, end = re_match.span()
            seq = text[start:end]
            prefix = ''
            suffix = ''
                    
            # Check before first punctuation        
            if start > 0 and text[start - 1] == ' ':
                # Skip extra spaces before
                while start > 1 and text[start - 2] == ' ':
                    start -= 1
            elif start > 0 and text[start - 1] != ' ':
                # Add space before
                prefix = ' '
                
            # Check after first punctuation
            while len(seq) > 1 and seq[1] == ' ':
                seq = seq[:1] + seq[2:]
                
            # Check before second punctuation
            while len(seq) > 2 and seq[-2] == ' ':
                seq = seq[:-2] + seq[-1]
            
            # Check after second punctuation        
            if end < len(text) and text[end] == ' ':
                # Skip extra spaces after
                while end + 1 < len(text) and text[end + 1] == ' ':
                    end += 1
            elif end < len(text) and text[end] != ' ' and text[end].isalpha():
                # Add space after
                suffix = ' '      
            
            # Remove extra spaces
            text = text[:start] + prefix + seq + suffix + text[end:]  
            
        # 2. Spaces around solo punctuation
        re_matches = list(re.finditer(SOLO_PUNCTUATION, text))
        re_matches.reverse()
        for re_match in re_matches:
            start, end = re_match.span()
            char = text[start:end]
            
            # Skip spaces before
            while start > 0 and text[start - 1] == ' ':
                start -= 1
              
            if end < len(text) and text[end] == ' ':
                # Skip extra spaces after first space
                while end < len(text) - 1 and text[end + 1] == ' ':
                    end += 1       
            elif end < len(text) - 1 and text[end] != ' ':
                # Check if a space should be added after
                if text[end - 1] == '.' and ((text[end].isdigit() and start > 0 and text[start - 1].isdigit()) or 
                                             # If before and after dot there is a digit
                                            (text[end].isupper() and start > 0 and text[start - 1].isupper())):
                                            # If before and after dot there is a uppercase letter
                    pass
                elif text[end].isalpha():
                    # Otherwise, an extra space is needed after
                    char += ' '
            
            # Remove extra spaces
            text = text[:start] + char + text[end:]  
        
        # 3. Remove double punctuation marks sequences
        re_matches = list(re.finditer(DOUBLE_PUNCTUATION, text))
        re_matches.reverse()
        for re_match in re_matches:
            pos = re_match.span()[0]
            # Remove initial dot
            text = text[:pos] + text[pos + 1:]
            
        # 4. Reduce many dots to just 3
        re_matches = list(re.finditer(TOO_MANY_DOTS, text))
        re_matches.reverse()
        for re_match in re_matches:
            start, end = re_match.span()
            # Remove extra dots
            text = text[:start] + '...' + text[end:]
            
        # 5. Reduce many spaces to just one
        re_matches = list(re.finditer(MULTIPLE_SPACES, text))
        re_matches.reverse()
        for re_match in re_matches:
            start, end = re_match.span()
            # Remove extra dots
            text = text[:start] + ' ' + text[end:]
          
        return text
    
    
    def clean_entity(self, ent):
        ''' Clean an entity'''
        
        ent = ent.strip('"\'')
    
        # TODO Check cand trebuie scoasa toata punctuatia
        # Remove punctuation 
    #     ent = re.sub('[' + string.punctuation + ']', '', ent)
        
        return ent
        
    def generalize_text(self, text, options=['lower', 'diacritics']):
        ''' Generalize a text for comparisons. '''
        
        gen_text = text
        
        if 'lower' in options:
            gen_text = gen_text.lower()
            
        if 'diacritics' in options:
            gen_text = unidecode.unidecode(gen_text)
            
        if 'punctuation' in options:
            gen_text = re.sub('[' + string.punctuation + ']', '', gen_text)
        
        return gen_text
    
    def find_generalized_matches(self, sub, text):
        ''' Find occurances of a substring using generlized texts. '''
        
        # Generalize texts
        gen_old = re.escape(self.generalize_text(sub))
        gen_tran = self.generalize_text(text)
                
        re_matches = list(re.finditer(gen_old, gen_tran))
            
        if re_matches:
            # If matches were found, replace them in reverse order
            re_matches.reverse()
            
        return re_matches
                

    def is_date(self, ent):
        ''' Check if an entity represents a date '''
        
        if ent.label_ != 'DATETIME':
            return False
        
        return any(char.isdigit() for char in ent.text)


    def get_action_type(self, action):
        ''' Identify the type of action '''
        
        cleanAction = self.clean_entity(action.lower())
        
        for key in KEYWORDS_INLOCUIESTE:
            if key in cleanAction:
                return ACTION_INLOCUIESTE
            
        for key in KEYWORDS_CITESTE:
            if key in cleanAction:
                return ACTION_CITESTE
            
        for key in KEYWORDS_PRELUNGESTE_CU:
            if key in cleanAction:
                return ACTION_PRELUNGESTE_CU
            
        for key in KEYWORDS_PRELUNGESTE_PANA:
            if key in cleanAction:
                return ACTION_PRELUNGESTE_PANA
            
        for key in KEYWORDS_ELIMINA:
            if key in cleanAction:
                return ACTION_ELIMINA
            
        for key in KEYWORDS_DEVINE:
            if key in cleanAction:
                return ACTION_DEVINE
            
        for key in KEYWORDS_PROROGA_PANA:
            if key in cleanAction:
                return ACTION_PROROGA_PANA
            
        for key in KEYWORDS_PROROGA_CU:
            if key in cleanAction:
                return ACTION_PROROGA_CU
            
        for key in KEYWORDS_MODIFICA:
            if key in cleanAction:
                return ACTION_MODIFICA
        
        if cleanAction == KEYWORD_CU:
            return ACTION_CU
        
        return ACTION_UNKNOWN
    
    
    def get_period_type(self, period):
        ''' Get the type of period: days, months, years. '''
        
        periodLow = period.lower()
        periodType = PERIOD_ERROR
        
        for key in KEYWORDS_ZILE:
            if key in periodLow:
                periodType = PERIOD_ZILE
            
        for key in KEYWORDS_LUNI:
            if key in periodLow:
                periodType = PERIOD_LUNI
        
        for key in KEYWORDS_ANI:
            if key in periodLow:
                periodType = PERIOD_ANI
                
        if periodType != PERIOD_ERROR:
            # Extract numbers from text
            numbers = [int(s) for s in periodLow.split() if s.isdigit()]
            
            if len(numbers) == 1:
                return periodType, numbers[0]
                
        return PERIOD_ERROR, None
    
    def is_same_action(self, action1, action2):
        ''' Detect if two actions actually refer to the same transformation. '''
        
        if action1 == action2:
            return True
        
        for action_group in SAME_ACTION_GROUPS:
            if action1 in action_group and action2 in action_group:
                # If both actions are in the same group
                return True
            
        return False    
    
    def group_actions_basic(self, doc):
        ''' Group Old's and New's for a text with a single action. '''
    
        entities = doc.ents
        olds = []
        news = []
        actions = []
        action_types = []
        
        error = NO_ERROR
        
        for ent in entities:            
            if ent.label_ == 'action':   
                actions.append(self.clean_entity(ent.text))
                            
            elif ent.label_ == 'old':
                olds.append(self.clean_entity(ent.text))
                
            else:
                news.append(self.clean_entity(ent.text))
              
        # Get action types
        action_types = [self.get_action_type(action) for action in actions]
            
        # Check number of actions
        if len(actions) > 1:
            
            if actions[1] in ACTION_PREPOSITIONS:
                # If the second word is one of the action preopositions, add it to the previous action
                action = actions[0] + ' ' + actions[1]
                
                # Check if next actions are the same preposition
                for j in range(2, len(actions)):
                    if actions[2] != actions[1]:
                        error = ERROR_MANY_ACTIONS   
                        
                if error == NO_ERROR:
                    actions = [action]            
            else:
                # Check if same action several times
                action_type1 = action_types[0]
                for action_type2 in action_types[1:]:
                    if not self.is_same_action(action_type1, action_type2):
                        error = ERROR_MANY_ACTIONS
                        break
                    
                if error != ERROR_MANY_ACTIONS:
                    actions = [actions[0]]
                
        elif len(actions) == 0:
            error = ERROR_NO_ACTION
                
        elif len(olds) + len(news) == 0 or (len(olds) + len(news) > 1 and len(olds) != len(news)):
            error = ERROR_NUM_OLD_NEW
            
        return error, actions, action_types, olds, news    
    
    
    def action_prelungeste_cu(self, pasiv, activ, action, olds, news):
        ''' Method for action Prelungeste cu '''
            
        actionApplied = False
        transformed = pasiv
           
        # Find dates in Pasiv
        pasivDoc = self.nlp_model(transformed)
        pasivEnts =  pasivDoc.ents
            
        pasivPeriods = []
        for ent in pasivEnts:
            if self.is_date(ent):
                # Get the types of periods and the numbers
                periodType, periodNumber = self.get_period_type(ent.text)
                    
                if periodType != PERIOD_ERROR:
                    # If a period was correctly extracted
                    pasivPeriods.append((periodType, periodNumber))
            
        activPeriods = []
        for ent in news:
            # Get the types of periods and the numbers
            periodType, periodNumber = self.get_period_type(ent)
                
            if periodType != PERIOD_ERROR:
                # If a period was correctly extracted
                activPeriods.append((periodType, periodNumber))
                    
        if len(pasivPeriods) == 1 and len(activPeriods) == 1:
            # If there is just one period to increase
            if pasivPeriods[0][0] == activPeriods[0][0]:
                # If periods in Old and New have the same type
                    
                # Add extra spaces to make them individual words
                newPeriod = pasivPeriods[0][1] + activPeriods[0][1]
                    
                transformed = transformed.replace(' ' + str(pasivPeriods[0][1]) + ' ', ' ' + str(newPeriod) + ' ')
                    
                actionApplied = True
                
        return transformed, actionApplied
    
    
    def action_prelungeste_pana(self, pasiv, activ, action, olds, news):
        ''' Method for action Prelungeste pana '''
            
        actionApplied = False
        transformed = pasiv
            
        # Find dates in Pasiv
        pasivDoc = self.nlp_model(transformed)
        pasivEnts =  pasivDoc.ents
            
        pasivDates = []
        for ent in pasivEnts:
            if self.is_date(ent):
                pasivDates.append(ent.text)
                            
        if len(pasivDates) == 1 and len(news) == 1:
            # If there is just one date in Pasiv and one New date
            transformed = transformed.replace(pasivDates[0], news[0])
                
            actionApplied = True
            
        return transformed, actionApplied
    
    
    def action_inlocuieste(self, pasiv, activ, action, olds, news):
        ''' Method for action Inlocuieste '''
            
        actionApplied = False
        transformed = pasiv
        
        if len(olds) + len(news) > 0 and len(olds) == len(news):
            # If there are corresponding Old's and New's            
            for i, old in enumerate(olds):
                
                # Find all generalized matches
                matches = self.find_generalized_matches(old, transformed)
                                  
                for match in matches:
                    start, end = match.span()
                    transformed = transformed[:start] + news[i] + transformed[end:]
                    actionApplied = True                    
    
        return transformed, actionApplied
    
    
    def action_elimina(self, pasiv, activ, action, olds, news):
        ''' Method for action Elimina '''
            
        actionApplied = False
        transformed = pasiv
        
        activPrefix = activ[activ.find(action) + len(action) + 1:]
        if (activPrefix.startswith('virgula dupa') or activPrefix.startswith('virgula după') or 
            activPrefix.startswith('virgulă după')) and len(olds) == 1:
                
            # Find all generalized matches
            matches = self.find_generalized_matches(olds[0], transformed)
                        
            for match in matches:
                start, end = match.span()
                # Skip over comma
                transformed = transformed[:start + len(olds[0])] + transformed[start + len(olds[0]) + 1:]
                actionApplied = True 
                
        elif (activPrefix.startswith('virgula inainte de') or activPrefix.startswith('virgula înainte de') or activPrefix.startswith('virgula dinainte de') or
              activPrefix.startswith('virgulă înainte de') or activPrefix.startswith('virgulă dinainte de')) and len(olds) == 1:
                  
            # Find all generalized matches
            matches = self.find_generalized_matches(olds[0], transformed)
            
            for match in matches:
                start, end = match.span()
                # Skip over comma
                transformed = transformed[:start - 2] + transformed[start - 1:]    
                actionApplied = True
            
        
        elif len(olds) > 0 and len(news) == 0:
            # If there is at least one Old and no News 
            
            for i, old in enumerate(olds):
                pos = transformed.find(old)
                if pos > -1:    
                    transformed = transformed[:pos] + transformed[(pos + 1) + len(old):]
                    
                    actionApplied = True
    
        return transformed, actionApplied
    
    
    def action_devine(self, pasiv, activ, action, olds, news):
        ''' Method for action Devine '''
            
        actionApplied = False
        transformed = pasiv
                    
        if len(olds) + len(news) > 0 and len(olds) == len(news):
            # If there are corresponding Old's and New's       
            transformed, actionApplied = self.action_inlocuieste(pasiv, activ, action, olds, news)
            
        elif len(news) == 1 and len(olds) < 2:
            # Check if the New is a date
            newEnts = self.nlp_model(news[0]).ents
            if len(newEnts) == 1 and self.is_date(newEnts[0]):
                # If there is just one New and it is a date, apply Prelungeste pana
                transformed, actionApplied = self.action_prelungeste_pana(pasiv, activ, action, olds, news)
    
        return transformed, actionApplied
    
    
    def transform(self, activ_ner, pasiv, activ):
        ''' Transform an instance of Pasiv using the corresponding instance of Activ. '''
        
        doc = activ_ner(activ)
            
        error, actions, action_types, olds, news = self.group_actions_basic(doc)
        
        if self.debug:
            print('Action:', actions)
            print('Olds:', olds)
            print('News:', news)
        
        if error != NO_ERROR:
            return error, False, "", actions, olds, news
        
        # Only one action identified
        action = actions[0]
        action_type = action_types[0]
        
        actionApplied = False
        transformed = pasiv
        
        if action_type == ACTION_CITESTE:
            # Apply Citeste
            transformed, actionApplied = self.action_inlocuieste(pasiv, activ, action, olds, news)
    
        elif action_type == ACTION_PRELUNGESTE_CU:
            # Apply Prelungeste cu
            transformed, actionApplied = self.action_prelungeste_cu(pasiv, activ, action, olds, news)
    
        elif action_type == ACTION_PRELUNGESTE_PANA:
            # Apply Prelungeste pana
            transformed, actionApplied = self.action_prelungeste_pana(pasiv, activ, action, olds, news)
                
        elif action_type == ACTION_INLOCUIESTE:   
            # Apply Inlocuieste
            transformed, actionApplied = self.action_inlocuieste(pasiv, activ, action, olds, news)
                
        elif action_type == ACTION_ELIMINA:  
            # Apply Elimina
            transformed, actionApplied = self.action_elimina(pasiv, activ, action, olds, news)
                
        elif action_type == ACTION_DEVINE:
            # Apply Devine
            transformed, actionApplied = self.action_devine(pasiv, activ, action, olds, news)
    
        elif action_type == ACTION_PROROGA_PANA:
            # Apply Prelungeste pana
            transformed, actionApplied = self.action_prelungeste_pana(pasiv, activ, action, olds, news)
    
        elif action_type == ACTION_PROROGA_CU:
            # Apply Prelungeste cu
            transformed, actionApplied = self.action_prelungeste_cu(pasiv, activ, action, olds, news)

        elif action_type == ACTION_CU:
            # Apply Inlocuieste cu
            transformed, actionApplied = self.action_inlocuieste(pasiv, activ, action, olds, news)

        elif action_type == ACTION_MODIFICA:
            # Apply Inlocuieste cu
            transformed, actionApplied = self.action_inlocuieste(pasiv, activ, action, olds, news)
    
        if actionApplied:
            # Clean any possible punctuation mistakes after the transformation
            transformed = self.clean_punctuation(transformed)
            
        if action_type == ACTION_UNKNOWN:
            error = ERROR_ACTION_UKNOWN
                                                       
        return error, actionApplied, transformed, actions, olds, news
    
    def transform_ensamble(self, pasiv, activ):
        ''' Apply the transform function for an ensamble of NERs. '''
        
        for ner in self.activ_ners:
            # Apply transform function for current NER
            error, actionApplied, transformed, actions, olds, news = self.transform(ner, pasiv, activ)
            
            if actionApplied == True:
                # If the action could be applied, return the results and do not apply other NERs
                break
            else:
                # Otherwise, continue to try the other NERs
                continue
          
        # Either return the correct results of a NER which applied the action or the error of the last NER
        return error, actionApplied, transformed, actions, olds, news
    
    
    ##############
    # Heuristics #
    ##############
    
    def match_keywords(self, activ, keywords):
        """ Search for keywords in the text. """
        
        match_dict = {}
        for key in keywords:
            pos = activ.find(key)
            
            if pos > -1:
                match_dict[pos] = key   
                
        return match_dict
    
    def search_modifica_cuprins(self, activ):
        """ Search for the phrase 'se modifică şi va avea următorul cuprins'. """
        
        match_dict = self.match_keywords(activ, MODIFICA_CUPRINS_KEYS)
        
        if match_dict:
            # If phrase found
            pos, _ = list(match_dict.items())[0]
            
            # Find first and last quotation mark
            q1 = activ.find('"', pos + 1)
            q2 = activ.rfind('"')
            transf = activ[q1 + 1:q2]
            
            return transf
        
        return None

    
    
    
    def _pre_process(self, inputs):
       
        self.debug = bool(inputs.get('DEBUG', False))
            
        self.apply_ensamble = bool(inputs.get('ENSAMBLE', True))
        
                
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
        activ = self.remove_links(activ).strip()        
    
        return pasiv, activ

    def _predict(self, prep_inputs):
        
        pasiv, activ = prep_inputs
        
        # First check heuristics
        transformed = self.search_modifica_cuprins(activ)
        if transformed:
            # If a transformation was made
            error = None
            actionApplied = True
            action = ["se modifică şi va avea următorul cuprins"]
            news = [transformed]
            olds = []
            return error, actionApplied, transformed, action, olds, news
        
        if self.apply_ensamble: 
            # Apply ensamble of NERs
            error, actionApplied, transformed, action, olds, news = self.transform_ensamble(pasiv, activ)
        else:
            # Apply single NER
            error, actionApplied, transformed, action, olds, news = self.transform(self.activ_ner, pasiv, activ)
        
        return error, actionApplied, transformed, action, olds, news

    def _post_process(self, pred):
        
        error, actionApplied, transformed, action, olds, news = pred
        
        res = {}
        
        res['action'] = action
        res['old'] = olds
        res['new'] = news
        
        success = False
        if error == ERROR_NO_ACTION:
            res['error'] = "No Action identified."
        elif error == ERROR_MANY_ACTIONS:
            res['error'] = "Too many Actions identified. Can only handle a single Action."
        elif error == ERROR_NUM_OLD_NEW:
            res['error'] = "Incorrect number of Old and New entities identified."
        elif error == ERROR_ACTION_UKNOWN:
            res['error'] = "Unknown action."
        elif actionApplied == False:
            # No action error, but action was not applied
            res['error'] = "Conditions for Old or New entities to apply Action not satisfied."
        else:
            # No error
            success = True
            res['result'] = transformed
                
        res['success'] = success
        
        return res


if __name__ == '__main__':
    from libraries import Logger
    
    l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
    eng = GetMergeV2Worker(log=l, default_config=_CONFIG, verbosity_level=1)
  
    test = {      
        # 'PASIV' : """Veniturile nete lunare prevăzute la alin. (1) şi (2) se majorează cu 5.000 lei pentru fiecare membru din familie care face dovada că lucrează în baza unui contract individual de muncă, a unei convenţii civile sau că realizează venituri din activităţi pe cont propriu.""",
        # 'ACTIV' : """Suma prevăzută la art. 3 alin. (3), la art. 4 alin. (2) şi la art. 12 alin. (2) din Legea nr. 67/1995 devine 5.300 lei.""",
        
        # se prelungeste cu
        # 'PASIV' : """Cererea de stabilire a dreptului de proprietate se introduce şi se înregistrează la primărie în termen de 30 de zile de la data intrării în vigoare a prezentei legi.""",
        # 'ACTIV' : """Termenul de 30 de zile prevăzut de art. 10 alin. 4 din Legea fondului funciar, ~id_link=920541;nr. 18/1991~, pentru introducerea şi înregistrarea cererii de stabilire a dreptului de proprietate se prelungeşte cu 15 zile.""",
        
        # se vor citi
        # 'PASIV' : """persoanele ale căror venituri sînt de pînă la 50.000 lei vor contribui cu o cotă de 2% din veniturile realizate, dar nu mai puţin de 2% din salariul mediu pe ţară.""",
        # 'ACTIV' : """În Ordonanţa Guvernului României ~id_link=918610;nr. 22/1992~, publicată în Monitorul Oficial al României, Partea I, nr. 213 din 28 august 1992, la articolul 4 alineatul 1 liniuţa a 2-a, cuvintele "salariul mediu pe ţară" se vor citi "salariul minim pe ţară".""",
                
        # se elimina
        # 'PASIV' : """unităţi fără personalitate juridică, cum sînt: oficiile teritoriale de difuzare a filmelor, cinematografe proprii în Bucureşti, Constanţa şi Piteşti, laboratorul pentru subtitrarea filmelor, realizarea copiilor de filme şi expedierea lor în reţeaua cinematografică, redacţia de publicaţii, tipografie şi şcoala de calificare a personalului.""",
        # 'ACTIV' : """la art. 2 lit. b) se elimină cuvintele: "... cinematografe proprii în Bucureşti, Constanţa şi Piteşti;""",
                
        # se prelungeste pana
        # 'PASIV' : """Raportul comisiei va fi depus pînă la data de 21 iunie 1993.""",
        # 'ACTIV' : """Termenul stabilit de art. 1 alin. 2 din Hotărârea Camerei Deputaţilor ~id_link=910062;nr. 69/1992~ privind încuviinţarea Comisiei pentru industrii şi servicii de a porni o anchetă, cu modificările ulterioare, se prelungeşte până la 15 octombrie 1993.""",
                
        # se inlocuieste
        # 'PASIV' : """Carne şi preparate din carne""",
        # 'ACTIV' : """Cu aceeaşi dată se abrogă alin. 2 al art. 1, poziţia 1. "Bovine (tineret şi adulte)" din anexa nr. 1, precum şi anexa nr. 2 la Hotărârea Guvernului nr. 197 bis din 30 aprilie 1993, iar poziţia "Carne şi preparate din carne" din anexa la Hotărârea Guvernului ~id_link=908888;nr. 206/1993~ se înlocuieşte cu poziţia "Carne de porcine şi de pasăre. Preparate din carne".""",
                
        # se inlocuieste
        # 'PASIV' : """Institutul Politehnic "Gheorghe Asachi" Iaşi""",
        # 'ACTIV' : """La litera A punctul 1 liniuţa a 19-a, denumirea Institutul Politehnic "Gheorghe Asachi" Iaşi se înlocuieşte cu denumirea Universitatea Tehnică "Gheorghe Asachi" Iaşi.""",
           
        # devine
        # 'PASIV' : """Dreptul Împrumutatului la trageri din suma disponibilă va înceta la 31 decembrie 1997 sau la o dată ulterioară stabilită de Bancă. Banca va înştiinţa prompt Împrumutatul asupra acestei date ulterioare.""",
        # 'ACTIV' : """Data specificată în secţiunea 2.03 din articolul 2 al acordului de împrumut se modifică şi devine 30 septembrie 1998.""",
          
        # se proroga pana
        # 'PASIV' : """Compania are obligaţia de a plăti obligaţiile restante prevăzute la alin. (2) până la data de 30 noiembrie 2012.""",
        # 'ACTIV' : """Termenele prevăzute la art. 1 alin. (1) şi (7) se prorogă până la 20 decembrie 2012 inclusiv.""",
          
        # devine
        # 'PASIV' : """Obligaţiunile de tezaur exprimate în dolari S.U.A. cu dobândă sunt scadente la data de 13 august 1997.""",
        # 'ACTIV' : """La punctul 3 - data scadenţei devine 15 august 1997.""",
        
        # cu
        # 'PASIV' : """În cazul în care poluarea s-a produs în zona economică exclusivă şi fondul nu se constituie voluntar, Ministerul Finanţelor formulează în faţa instanţei judecătoreşti competente, în numele statului român, cererea de compensare a pagubelor produse prin poluare.""",
        # 'ACTIV' : """"Ministerul Finanţelor" cu "Ministerul Finanţelor Publice".""",
        
        # se inlocuieste cu
        # 'PASIV' : """'A se vedea şi 6.2.1.7.'""",
        # 'ACTIV' : """"La articolul 5.2.1.6 în cadrul NOTEI se înlocuieşte '6.2.1.7' cu '6.2.2.7' şi '6.2.1.8' cu '6.2.2.8'.""",
        
        # se modifică şi va avea următorul cuprins        
        # 'PASIV' : """(3) Cifra de şcolarizare pentru rezidenţiat este cel puţin egală cu numărul de locuri reprezentând totalul absolvenţilor de medicină, medicină dentară şi farmacie cu diplomă de licenţă din promoţia anului în curs, cumulat cu numărul de posturi conform art. 18, stabilită prin ordin al ministrului sănătăţii. În cazul în care numărul candidaţilor pentru domeniul medicină care promovează examenul de rezidenţiat este mai mare decât cifra de şcolarizare iniţial anunţată, aceasta se poate suplimenta până la repartiţia candidaţilor, astfel încât toţi candidaţii promovaţi să poată accesa un loc sau un post de rezidenţiat. Ministerul Finanţelor asigură resursele financiare necesare şcolarizării prin rezidenţiat la nivelul cifrelor de şcolarizare aprobate.  """,
        # 'ACTIV' : """   1. La articolul 2, alineatul (3) se modifică şi va avea următorul cuprins: " (3) Cifra de şcolarizare pentru rezidenţiat este stabilită anual prin ordin al ministrului sănătăţii. La stabilirea cifrei de şcolarizare se are în vedere capacitatea de pregătire disponibilă comunicată de instituţiile de învăţământ superior cu profil medical acreditate, până cel târziu la data de 1 august a fiecărui an. Pentru domeniul medicină, cifra de şcolarizare este cel puţin egală cu numărul absolvenţilor cu diplomă de licenţă din promoţia anului în curs." """,
        
        
        # 'PASIV' : """art. 4   (4) În cazul în care se constată că beneficiarii nu au respectat criteriile de eligibilitate şi angajamentele prevăzute în ghidurile de finanţare care constituie Programul \"ELECTRIC UP\" privind finanţarea întreprinderilor mici şi mijlocii pentru instalarea sistemelor de panouri fotovoltaice pentru producerea de energie electrică şi a staţiilor de reîncărcare pentru vehicule electrice şi electrice hibrid plug-in, au făcut declaraţii incomplete sau neconforme cu realitatea pentru a obţine ajutorul de minimis sau au schimbat destinaţia acestuia ori se constată că nu au respectat obligaţiile prevăzute în contractul de finanţare, se recuperează, potrivit dreptului comun în materie, ajutorul de minimis acordat, cu respectarea normelor naţionale şi europene în materia ajutorului de stat de către Ministerul Economiei, Energiei şi Mediului de Afaceri în calitate de furnizor.""",
        # 'ACTIV' : """În cuprinsul Ordonanţei de urgenţă a Guvernului nr. 159/2020 privind finanţarea întreprinderilor mici şi mijlocii şi domeniului HORECA pentru instalarea sistemelor de panouri fotovoltaice pentru producerea de energie electrică cu o putere instalată cuprinsă între 27 kWp şi 100 kWp necesară consumului propriu şi livrarea surplusului în Sistemul energetic naţional, precum şi a staţiilor de reîncărcare de minimum 22 kW pentru vehicule electrice şi electrice hibrid plug-in, prin Programul de finanţare \"ELECTRIC UP\", sintagma \"Ministerul Economiei, Energiei şi Mediului de Afaceri\" se înlocuieşte cu sintagma \"Ministerul Energiei\".""",

        # test 1 client
        # 'PASIV' : """Modelul-cadru al atestatului de persoană sau familie aptă să adopte, precum şi modelul şi conţinutul unor formulare, instrumente şi documente utilizate în procedura adopţiei se aprobă prin ordin a preşedintelui Oficiului.""",
        # 'ACTIV' : """În cuprinsul Legii nr. 273/2004 privind procedura adopției, republicată, cu modificările și completările ulterioare, precum și în cuprinsul actelor normative în vigoare din domeniul adopției, sintagma "deschiderea procedurii adopției interne" se înlocuiește cu sintagma "deschiderea procedurii adopției", denumirea "Oficiul" se înlocuiește cu denumirea "A.N.P.D.C.A.", sintagma "potrivire teoretică" se înlocuiește cu sintagma "potrivire inițială" și termenul "ordin" se înlocuiește cu termenul "decizie".""",
        
        # test 2 client
        # 'PASIV' : """Dacă, în urma admiterii acțiunii, autoritatea administrativă este obligată să înlocuiască sau să modifice actul administrativ, să elibereze un certificat, o adeverință sau orice alt înscris, executarea hotărârii definitive se va face în termenul prevăzut în cuprinsul ei, iar în lipsa unui astfel de termen, în cel mult 30 de zile de la data rămînerii definitive a hotărîrii.""",
        # 'ACTIV' : """La articolul 16 alineatul 1, noțiunea "hotarârea definitivă" se înlocuiește cu "hotărârea irevocabilă".""",
         
        # test 3 client
        'PASIV' : """Fondul Proprietății de Stat şi celelalte instituţii publice abilitate să efectueze operaţiuni în cadrul procesului de restructurare şi privatizare au obligaţia să deruleze fondurile rezultate, prin conturi deschise la trezoreria statului.""",
        'ACTIV' : """În tot cuprinsul ordonanței de urgență sintagma Fondul Proprietății de Stat se înlocuiește cu sintagma Autoritatea pentru Privatizare şi Administrarea Participaţiilor Statului.""",
        
        'ENSAMBLE': True,
        'DEBUG': True
      }
          
    res = eng.execute(inputs=test, counter=1)
    print(res)
    

    
    
    ################
    # Client tests #
    ################    
    
        
    # testsFile = "C:\\Proiecte\\LegeAI\\Date\\Task9\\teste_client\\minim 5 exemple cu diferite acţiuni_mod.xlsx"
    # NUM_ACTIV = 3
    # NUM_PASIV = 4
    # NUM_TRANSF = 5
        
    # testSheets = pd.read_excel(testsFile, sheet_name=None)
    
    # nCor, nIncor, nFail = 0, 0, 0
    
    # for sheet in testSheets:
    #     sheetDf = testSheets[sheet]
        
    #     nCorSh, nIncorSh, nFailSh = 0, 0, 0
            
    #     # No NaN values in Pasiv, Activ or Transf
    #     sheetDf = sheetDf[~sheetDf.iloc[:, [NUM_ACTIV, 
    #                                         NUM_PASIV, 
    #                                         NUM_TRANSF]].isnull().any(axis=1)]
            
    #     for i, row in sheetDf.iterrows():
    #         pasiv = row[NUM_PASIV]
    #         activ = row[NUM_ACTIV]
    #         transf = row[NUM_TRANSF]
            
    #         test = {                 
    #             'PASIV' : pasiv,
    #             'ACTIV' : activ,
    #             'ENSAMBLE': True,
    #             'DEBUG': False
    #           }
            
    #         res = eng.execute(inputs=test, counter=i)
                
    #         if not 'success' in res:
    #             # print('Error', i)
    #             continue
    #         else: 
    #             if res['success'] == True:
    #                 # print(res)
    #                 if res['result'].strip(" \n\t") == transf.strip(" \n\t"):
    #                     # print('Correct')
    #                     nCor += 1
    #                     nCorSh += 1
    #                 else:
    #                     # print('Incorrect')
    #                     nIncor += 1
    #                     nIncorSh += 1
    #             else:
    #                 # print(res)
    #                 # print('Fail')
    #                 nFail += 1
    #                 nFailSh += 1
        
    #     total = nCorSh + nIncorSh + nFailSh
    #     print('{} - {} / {}; {} incorrect'.format(sheet, nCorSh, total, nIncorSh))
        
    # total = nCor + nIncor + nFail
    # print('Total: {} / {}, {:.2f}%'.format(nCor, total, nCor * 100 / total))