import pickle as pk


if __name__ == '__main__':
  path_in = [
    './_cache/_data/20211203_154548_x_data',
    './_cache/_data/20211203_154548_y_data',
    './_cache/_data/dict_lbl',
    './_cache/_data/embgen_full_embeds'
    './_cache/_data/lai_ro_i2w_191K',
    './_cache/_data/lai_embeddings_191K'
    './_cache/_data/20211203_154548_labels_dict'
  ]

  for path in path_in:
    print('Loading {}'.format(path))
    with open(path + '.pkl', 'rb') as f:
      data = pk.load(f)

    print('Saving {}'.format(path))
    with open(path + "_37.pkl", 'wb') as f:
      pk.dump(f, data, protocol=4)

  print("Convering done")