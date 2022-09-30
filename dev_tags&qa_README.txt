Readme for tags&qa dev

####################################################
Corpora:

tags_v1: original tags
train: 131.510, dev: 16.438, test: 16.440, total: 164.388

qa_v3: 29.03.2022 from new db legeV
tags_title: tags_v1 with only titles, corpora extracted on 05.04.2022
train: 83.133, dev: 10.391, test: 10.393, total: 103.917

####################################################

Endpoints:

get_tags: model trained on corpus tags_v1
get_tags_orig: original get tags model

get_qa: model trained on tags_title_v1 with tags labels
get_qa_orig: original, not mine

####################################################

How to run:

#######################

    train tagger from scratch
python tagger\brain\train_transformer_tagger.py -use_generator -data_path=_cache/_data/tags_v1 -k 1 3 5 -epochs=5 -learning_rate=1e-5 -bert_max_seq_len=128 -batch_size=32 -bert_backbone=readerbench/jurBERT-base -run_type=train -model_path=_cache/_models/0

#######################

    train qa from tagger
    on qa:
python tagger\brain\train_transformer_tagger.py -data_path=_cache/_data/qa_v2 -k 1 3 -epochs=5 -learning_rate=1e-3 -bert_frozen -bert_max_seq_len=128 -batch_size=4 -bert_backbone=_cache/_models/tags_5/05 -run_type=train -model_path=_cache/_models/test
    on tags_title:
python tagger\brain\train_transformer_tagger.py -use_generator -data_path=_cache/_data/tags_titles_v1 -k 1 3 5 -epochs=5 -learning_rate=1e-5 -bert_max_seq_len=64 -batch_size=32 -bert_backbone=_cache/_models/finetuned_tags_bert_512 -run_type=train -model_path=_cache/_models/tags_titles_1

#######################

    evaluate model
python tagger\brain\train_transformer_tagger.py -data_path=_cache/_data/qa_v1 -k 1 3 -learning_rate=1e-3 -bert_max_seq_len=128 -batch_size=4 -run_type=eval -model_path=_cache/_models/qa_1/weights/05 -bert_backbone=_cache/_models/qa_1/05
python tagger/brain/train_transformer_tagger.py -args_path=_cache/_models/qa_1/eval_config.json

#######################

get stats
from live (does not work for tags): python -W ignore get_stats.py
from saved data (plotting might take a while) python get_stats.py -data_path=_cache/_data/qa_vX

####################################################

Misc


Env:
- added transformers==4.10.2 to conda env
