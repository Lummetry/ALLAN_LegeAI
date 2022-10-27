TRAIN_LEN = 5035

import sys
import os
import json
import spacy
import random
import numpy as np
from spacy.tokens import DocBin
from spacy.cli.train import train
import shutil



print('Loaded spacy model.')

MAX_NERS = 8
MIN_NEW_ANNS = 100

# relative paths to the root of the repository
configPath = "./training/merge/config.cfg"
trainPath = "./training/merge/train.spacy"
devPath = "./training/merge/dev.spacy"

# absolute paths
train_folder_path = 'C:/Users/damian/Desktop/TRAIN_GET_MERGE_DATA'
ner_folder = 'C:/allan_data/MergeNER'


def parseActivs(filePath):
    ''' 
    Parse the Activ texts and labels from a file. 
    Also return the list of unlabelled texts.
    '''
    
    with open(filePath, 'r', encoding='utf8') as jsonFile:
        jsonList = list(jsonFile)
        
    annotations = []    
    unlabelled = []

    for jsonStr in jsonList:
        line = json.loads(jsonStr)

        labels = []
        if line['label']:
            labels = line['label']

            i = 0
            while i < len(labels) - 1:

                label1 = labels[i]
                label2 = labels[i + 1]

                if label1[2] == label2[2] and label2[0] - label1[1] <= 2:
                    # Same type and close adjacent

                    # Unite the labels
                    label1[1] = label2[1]

                    # Delete the next label
                    labels = labels[:i + 1] + labels[i + 2:]
                else:    
                    i += 1

            annotations.append({'text' : line['text'],
                           'labels' : labels})
        else:
            unlabelled.append(line)
    
    return annotations, unlabelled


def dataToDocBin(annotations, path):

    nlp = spacy.blank("ro")
    db = DocBin()

    for ann in annotations:
        doc = nlp(ann['text'])
        labels = ann['labels']
        ents = []

        for start, end, label in labels:
            span = doc.char_span(start, end, label=label)
            ents.append(span)

        try:
            doc.ents = ents
            db.add(doc)
        except:
            continue

    db.to_disk(path)


if __name__ == '__main__':
    
    # Read folder paths   
    path1 = input('Path train data folder [Default = {}]:'.format(train_folder_path))
    path2 = input('Path NER folder [Default = {}]:'.format(ner_folder))
    if path1 != '':
        train_folder_path = path1
    if path2 != '':
        ner_folder = path2
            
    
    # Read all jsonl files
    anns = []
    
    for file in os.listdir(train_folder_path):
        if file.endswith(".jsonl"):
            train_file_path = os.path.join(train_folder_path, file)
            
            # Parse file
            newAnns, unlabelled = parseActivs(train_file_path)
            anns.extend(newAnns)
    
    print('Read {} annotations.'.format(len(anns)))
    
    if abs(len(anns) - TRAIN_LEN) < MIN_NEW_ANNS:
        print('Too few new annotations. Must be at least {}.'.format(MIN_NEW_ANNS))
        
    else:
    
        # Train-test-dev split
        nDev = 150
        nTest = 150
        
        idxs = list(range(len(anns)))
        random.seed(42)
        random.shuffle(idxs)
        
        anns = np.array(anns)
        trainAnn = anns[idxs[:-(nDev + nTest)]]
        devAnn = anns[idxs[-(nDev + nTest):-nTest]]
        testAnn = anns[idxs[-nTest:]]
        
        print('Split: Train {} / Test {} / Dev {}'.format(len(trainAnn), len(devAnn), len(testAnn)))
        
        
        # Transform data
        dataToDocBin(trainAnn, trainPath)
        dataToDocBin(devAnn, devPath)
        
        
        # Check model folder
        min_timestamp = -1
        oldest_model = ''
        n = 0
        for file in os.listdir(ner_folder):
            ner_file_path = os.path.join(ner_folder, file)
            timestamp = os.path.getmtime(ner_file_path) 
            
            # Ensure the models are not Read Only    
            os.chmod(ner_file_path, 0o777)
            
            if min_timestamp == -1 or timestamp < min_timestamp:
                min_timestamp = timestamp
                oldest_model = ner_file_path
            n += 1
        
        # If there are too many models
        if n >= MAX_NERS - 1:    
            try:
                # Delete the oldest model
                shutil.rmtree(oldest_model)
            except Exception:
                pass
        
        
        # Train model
        train(configPath, ner_folder, overrides={"paths.train": trainPath, "paths.dev": devPath})
        
        
        # Edit training results
        try:
            # Remove model-last
            model_last_folder = os.path.join(ner_folder, 'model-last')
            shutil.rmtree(model_last_folder)
        except Exception:
            pass
        
        try:
            # Rename model-best
            model_best_folder = os.path.join(ner_folder, 'model-best')
            model_name = os.path.join(ner_folder, 'model-{}-{}-{}'.format(len(trainAnn), len(devAnn), len(testAnn)))
            os.rename(model_best_folder, model_name)
        except Exception:
            pass
        
        
        
        
        # Update constant at the beggining of the script
        
        with open(__file__, 'r') as f:
            lines = f.read().split('\n')
            val = int(lines[0].split(' = ')[-1])
            new_line = 'TRAIN_LEN = {}'.format(len(anns))
            new_file = '\n'.join([new_line] + lines[1:])
        with open(__file__, 'w') as f:
            f.write('\n'.join([new_line] + lines[1:]))