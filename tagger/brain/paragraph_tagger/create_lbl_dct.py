from libraries import Logger


if __name__ == '__main__':
  EMB_MODEL_NAME = 'test_model'
  DATA_SUBFOLDER_PATH = 'tagger_dataset'
  # DATA_MAPPER_FN = '{}/data_mapper.json'.format(DATA_SUBFOLDER_PATH)
  DCT_LBL_FN = 'dict_lbl.pk'
  EMBEDS_FN = ''
  LOGGER_CONFIG = 'tagger/brain/configs/20211202/config_train.txt'
  MODELS_DEF_FN = 'tagger/brain/configs/20211202/models_defs.json'

  FIXED_LENGTH = 500
  BATCH_SIZE = 512
  NR_EPOCHS = 30


  log = Logger(lib_name='TRN', config_file=LOGGER_CONFIG)

  y = log.load_pickle_from_data("20211202_220937_y_data_19K.pkl")
  y_set = set()
  for _y in y:
    y_set = y_set.union(set(_y))
  dct_y = {v: k for k, v in enumerate(y_set)}

  log.save_pickle_to_data(dct_y, '20211203_154548_y_data.pk')