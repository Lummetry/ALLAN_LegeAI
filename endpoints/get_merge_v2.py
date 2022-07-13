# -*- coding: utf-8 -*-

from libraries.model_server_v2 import FlaskWorker
import regex as re
import spacy

_CONFIG = {
  'SPACY_MODEL' : 'ro_core_news_lg',
 }

# File paths
# Debug
NER_MODEL_DEBUG = 'C:\\Proiecte\\LegeAI\\Date\\Task9\\output\\model-best'
# Prod
NER_MODEL_PROD = 'C:\\allan_data\\...\\model-best'


MIN_PASIV_WORDS = 2
MIN_ACTIV_WORDS = 5

# Actions
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


__VER__='0.1.0.0'
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
    
    
    def group_actions_basic(self, doc):
        ''' Group Old's and New's for a text with a single action. '''
    
        entities = doc.ents
        olds = []
        news = []
        action = None
        
        i = 0
        while i < len(entities):
            ent = entities[i]
            
            if ent.label_ == 'action':
                action = ent.text
                
                for j in range(i + 1, len(entities)):
                    nextEnt = entities[j]
                    
                    if nextEnt.label_ == 'action':
                        # If there is another action                    
                        if len(nextEnt) > 1:
                            return -1, None, None
                        
                        tok = doc[nextEnt.start]
                        if tok.is_stop:
                            action += " " + tok.text
                            i += 1
                        else:
                            return -1, None, None
                        
            elif ent.label_ == 'old':
                olds.append(self.clean_entity(ent.text))
            
            else:
                news.append(self.clean_entity(ent.text))
            
            i += 1
            
        if len(olds) + len(news) == 0 or (len(olds) + len(news) > 1 and len(olds) != len(news)):
            return -2, None, None
            
        return action, olds, news    
    
    
    def prelungeste_cu(self, pasiv, activ, action, olds, news):
        ''' Method for action Prelungeste cu '''
            
        actionApplied = False
        transformed = pasiv
           
        # Find dates in Pasiv
        pasivDoc = self.nlp_model(transformed)
        pasivEnts =  pasivDoc.ents
            
        pasivPeriods = []
        for ent in pasivEnts:
            if ent.label_ == 'DATETIME':
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
    
    
    def prelungeste_pana(self, pasiv, activ, action, olds, news):
        ''' Method for action Prelungeste pana '''
            
        actionApplied = False
        transformed = pasiv
            
        # Find dates in Pasiv
        pasivDoc = self.nlp_model(transformed)
        pasivEnts =  pasivDoc.ents
            
        pasivDates = []
        for ent in pasivEnts:
            if ent.label_ == 'DATETIME':
                pasivDates.append(ent.text)
            
        if len(pasivDates) == 1 and len(news) == 1:
            # If there is just one date in Pasiv and one New date
            transformed = transformed.replace(pasivDates[0], news[0])
                
            actionApplied = True
            
        return transformed, actionApplied
    
    
    def transform(self, pasiv, activ):
        ''' Transform an instance of Pasiv using the corresponding instance of Activ. '''
        
        doc = self.activ_ner(activ)
            
        action, olds, news = self.group_actions_basic(doc)
        
        if self.debug:
            print('Action:', action)
            print('Olds:', olds)
            print('News:', news)
        
        if action == None:
    #         print('No entities identified.')
            return None, None, None
        elif action == -1:
    #         print('Too many actions.')
            return None, None, None
        elif action == -2:
    #         print('NER error.')
            return None, None, None
    
        actionType = self.get_action_type(action)
        
        actionApplied = False
        transformed = pasiv
        
        if actionType == ACTION_CITESTE:
            
            if len(olds) + len(news) > 0 and len(olds) == len(news):
                # If there are corresponding Old's and New's
                
                for i, old in enumerate(olds):
                    pos = transformed.find(old)
                    if pos > -1:    
                        transformed = transformed.replace(old, news[i])
                        
                        actionApplied = True
    
        elif actionType == ACTION_PRELUNGESTE_CU:
            # Apply Prelungeste cu
            transformed, actionApplied = self.prelungeste_cu(pasiv, activ, action, olds, news)
    
        elif actionType == ACTION_PRELUNGESTE_PANA:
            # Apply Prelungeste pana
            transformed, actionApplied = self.prelungeste_pana(pasiv, activ, action, olds, news)
                
        elif actionType == ACTION_INLOCUIESTE:        
            
            if len(olds) + len(news) > 0 and len(olds) == len(news):
                # If there are corresponding Old's and New's            
                for i, old in enumerate(olds):
                    pos = transformed.find(old)
                    if pos > -1:    
                        transformed = transformed.replace(old, news[i])
                        
                        actionApplied = True
                
        elif actionType == ACTION_ELIMINA:        
            
            activPrefix = activ[activ.find(action) + len(action) + 1:]
            if (activPrefix.startswith('virgula dupa') or activPrefix.startswith('virgula după')) and len(olds) == 1:
                # Special case - virgula dupa
                pos = transformed.find(olds[0])
                if pos > -1:    
                    # Skip over comma
                    transformed = transformed[:pos + len(olds[0])] + transformed[pos + len(olds[0]) + 1:]                    
                    actionApplied = True
                    
            elif (activPrefix.startswith('inainte de') or activPrefix.startswith('înainte de') or activPrefix.startswith('dinainte de')) and len(olds) == 1:
                # Special case - virgula inainte
                pos = transformed.find(olds[0])
                if pos > -1:    
                    # Skip over comma
                    transformed = transformed[:pos - 2] + transformed[pos - 1:]                    
                    actionApplied = True
                
            
            elif len(olds) > 0 and len(news) == 0:
                # If there is at least one Old and no News 
                
                for i, old in enumerate(olds):
                    pos = transformed.find(old)
                    if pos > -1:    
                        transformed = transformed[:pos] + transformed[(pos + 1) + len(old):]
                        
                        actionApplied = True
                
        elif actionType == ACTION_DEVINE:
                    
            if len(olds) + len(news) > 0 and len(olds) == len(news):
                # If there are corresponding Old's and New's            
                for i, old in enumerate(olds):
                    pos = transformed.find(old)
                    if pos > -1:    
                        transformed = transformed.replace(old, news[i])
                        
                        actionApplied = True
            
            elif len(news) == 1 and len(olds) < 2:
                # Check if the New is a date
                newEnts = self.nlp_model(news[0]).ents
                if len(newEnts) == 1 and newEnts[0].label_ == 'DATETIME':
                    # If there is just one New and it is a date, apply Prelungeste pana
                    transformed, actionApplied = self.prelungeste_pana(pasiv, activ, action, olds, news)
    
        elif actionType == ACTION_PROROGA_PANA:
            # Apply Prelungeste pana
            transformed, actionApplied = self.prelungeste_pana(pasiv, activ, action, olds, news)
    
        elif actionType == ACTION_PROROGA_CU:
            # Apply Prelungeste cu
            transformed, actionApplied = self.prelungeste_pana(pasiv, activ, action, olds, news)
    
        if actionApplied:
            # Clean any possible punctuation mistakes after the transformation
            transformed = self.clean_punctuation(transformed)
                                                       
        return actionType, actionApplied, transformed
    
    
    
    def _pre_process(self, inputs):
       
        self.debug = bool(inputs.get('DEBUG', False))
        
        
        # Set paths
        if self.debug:
            ner_model_path = NER_MODEL_DEBUG
        else:
            ner_model_path = NER_MODEL_PROD            
        
        # Load trained NER model for Activ        
        self.activ_ner = spacy.load(ner_model_path)
                
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
        
        actionType, actionApplied, transformed = self.transform(pasiv, activ) 
              
        return transformed

    def _post_process(self, pred):
        
        transf = pred
        
        res = {}
        
        res['results'] = transf
        
        return res


if __name__ == '__main__':
    from libraries import Logger
    
    l = Logger('GESI', base_folder='.', app_folder='_cache', TF_KERAS=False)
    eng = GetMergeV2Worker(log=l, default_config=_CONFIG, verbosity_level=1)
  
    test = {      
        # 'PASIV' : """Veniturile nete lunare prevăzute la alin. (1) şi (2) se majorează cu 5.000 lei pentru fiecare membru din familie care face dovada că lucrează în baza unui contract individual de muncă, a unei convenţii civile sau că realizează venituri din activităţi pe cont propriu.""",
        # 'ACTIV' : """Suma prevăzută la art. 3 alin. (3), la art. 4 alin. (2) şi la art. 12 alin. (2) din Legea nr. 67/1995 devine 5.300 lei.""",
        
        'PASIV' : """Cererea de stabilire a dreptului de proprietate se introduce şi se înregistrează la primărie în termen de 30 de zile de la data intrării în vigoare a prezentei legi.""",
        'ACTIV' : """Termenul de 30 de zile prevăzut de art. 10 alin. 4 din Legea fondului funciar, ~id_link=920541;nr. 18/1991~, pentru introducerea şi înregistrarea cererii de stabilire a dreptului de proprietate se prelungeşte cu 15 zile.""",
        
        'DEBUG': True
      }
          
    res = eng.execute(inputs=test, counter=1)
    print(res)