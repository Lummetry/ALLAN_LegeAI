import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import Counter

parser = argparse.ArgumentParser(description='Get data stats for corpora')
parser.add_argument('-data_path', help='path + files desciptor (i.e. _cache/_data/qa_v1). if this parameters is missing, data is directly collected from live db')
args = parser.parse_args()


def find_values_by_occurences(counter, occurences):
  values = []

  for x in counter:
    if counter[x] == occurences:
      values.append(x)

  return values


def get_data_db():
  from libraries.db_conn.odbc_conn import ODBCConnector
  from libraries import Logger
  from collections import deque
  import time
  from utils.utils import raw_text_to_words, clean_words_list


  log = Logger(
    lib_name='DBSV', base_folder='.', app_folder='_cache',
    TF_KERAS=False
  )


  config = {
    'CONNECT_PARAMS' : {
      'DRIVER' : '{ODBC Driver 17 for SQL Server}',
      'SERVER' : '195.60.78.50',
      'PORT' : 1433,
      'DATABASE' : 'LegeV_New',
      'Uid' : 'damian',
      'Pwd' : '4Esoft1234!@#$2021',
    },

    'QUERY_PARAMS' : None
  }
  
  qry_pars = 'select distinct id_paragraf from paragraf_x_cuvant_cheie'
  qry_txt = 'select continut from paragraf where id={}'
  qry_lbl = """
  select nume from 
  tip_cuvantcheie, paragraf_x_cuvant_cheie
  where uid = ID_CUVANT_CHEIE and ID_PARAGRAF={}  
  """

  conn = ODBCConnector(log=log, verbose=False, config=config)
  conn.connect(nr_retries=5)
  
  df_pars = conn.get_data(sql_query=qry_pars)

  lst_X_pars = []
  lst_y_labels = []
  unique_labels = set()
  DEBUG = len(sys.argv) > 1 and sys.argv[1].upper() == 'DEBUG'
  log.P("Running params: {}. Debug mode {}".format(sys.argv, "ON" if DEBUG else "OFF"))
  n_iters = df_pars.shape[0]
  timings = deque(maxlen=10)
  for idx_par in range(n_iters):
    t0 = time.time()
    id_par = df_pars.iloc[idx_par,0]

    
    # process text
    df_text = conn.get_data(sql_query=qry_txt.format(id_par))
    if df_text.shape[0] > 1:
        print("More than one entry for par", id_par)
        sys.exit()
    elif df_text.shape[0] == 0:
        continue
    txt = df_text.iloc[0, 0]
    par_str = raw_text_to_words(txt, max_len=15)    
    if len(par_str) < 1:
        continue

    # process labels
    df_labels = conn.get_data(sql_query=qry_lbl.format(id_par))

    lst_raw_labels = [df_labels.iloc[iii, 0] for iii in range(df_labels.shape[0])]
    lst_labels = clean_words_list(lst_raw_labels)

    for lbl in lst_labels:
      unique_labels.add(lbl)

    lst_X_pars.append(par_str)
    lst_y_labels.append(lst_labels)

    lap_time = time.time() - t0
    timings.append(lap_time)
    mean_time = np.mean(timings)
    remaining_time = (n_iters - (idx_par + 1)) * mean_time
    if (idx_par % 10) == 0:
      print("\rProcessed {}/{} documents ({:.1f}%). Remaining time {:.0f}s/{} ({:.1f}s/doc\r".format(
        idx_par+1, n_iters, 
        (idx_par+1) / df_pars.shape[0] * 100, 
        remaining_time,
        time.strftime("%H:%M:%S", time.gmtime(remaining_time)),
        mean_time
        ),
        end='', flush=True)    

  return lst_X_pars, lst_y_labels  

if __name__ == "__main__":

    if args.data_path != None:
      docs = pickle.load(open(args.data_path + "_x_data.pkl", "rb"))
      labels = pickle.load(open(args.data_path + "_y_data.pkl", "rb"))
    else:
      docs, labels = get_data_db()
      print()

    print("Total number of documents:", len(docs))

    print("#"*100)

    words = []
    for doc in docs:
      words.extend(doc)
    lens = [len(x) for x in docs]
    words_counter = Counter(words)
    print("Total number of words {0} | Unique words {1}".format(len(words), len(words_counter)))
    print("Words per entry: Min {0} | Median {2} | Mean {1} | Max {3}".format(np.min(lens), np.mean(lens), np.median(lens), np.max(lens)))
    print("Most common 20 words:", words_counter.most_common(20))

    bc = np.bincount(lens)
    plt.bar(range(len(bc)), height=bc)
    plt.title('Distribution of number of words per document')
    plt.xlabel('no words')
    plt.ylabel('no documents')
    plt.show()


    print("#"*100)

    all_labels = []
    for x in labels:
        all_labels.extend(x)
    print("Total number of adnotations:", len(all_labels))

    labels_counter = Counter(all_labels)
    print("Total number of unique labels:", len(labels_counter))

    occurences = list(map(lambda x: labels_counter[x], labels_counter))

    print()
    bc = np.bincount(occurences)
    for index, value in reversed(list(enumerate(bc))):
      if value != 0:
        words = find_values_by_occurences(labels_counter, index)
        if value == 1:
          print("    {0} label appears {1} times: {2}".format(value, index, words[0]))
        elif index == 1:
          print("    {0} labels appear {1} time: {2}".format(value, index, ', '.join(words)))
        else:
          print("    {0} labels appear {1} times: {2}".format(value, index, ', '.join(words)))

        print()

    plt.hist(occurences, density=False, bins=range(max(occurences)+2))
    plt.title('Count of labels (OY) that appear n (OX) times')
    plt.ylabel('no labels')
    plt.xlabel('occurences')
    plt.xticks(range(max(occurences)+2), range(max(occurences)+2), rotation=90)
    plt.show()

    t = []
    for x in labels_counter:
      t.append(x)
    plt.bar(range(len(occurences)), height=occurences)
    plt.title('Occurences for each label')
    plt.xticks(range(len(occurences)), t, rotation=90)
    plt.ylabel('occurences')
    plt.subplots_adjust(bottom=0.4)
    plt.show()
