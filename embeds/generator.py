# -*- coding: utf-8 -*-
"""
Copyright 2019-2021 Lummetry.AI (Knowledge Investment Group SRL). All Rights Reserved.


* NOTICE:  All information contained herein is, and remains
* the property of Knowledge Investment Group SRL.  
* The intellectual and technical concepts contained
* herein are proprietary to Knowledge Investment Group SRL
* and may be covered by Romanian and Foreign Patents,
* patents in process, and are protected by trade secret or copyright law.
* Dissemination of this information or reproduction of this material
* is strictly forbidden unless prior written permission is obtained
* from Knowledge Investment Group SRL.


@copyright: Lummetry.AI
@author: Lummetry.AI
@project: 
@description:
@created on: Sat Nov 20 08:40:40 2021
@created by: damian


"""

import re

from gensim.test.utils import datapath
from gensim import utils

class CorpusGenerator:
  def __init__(self, path):
    self._datapath = path
    self.tag_cleaner = re.compile('<.*?>')
    return

  
  def remove_exclusions(self, line):
    clean_line = re.sub(self.tag_cleaner, '', line)
    return clean_line
    
    
  def __iter__(self):
    corpus_path = datapath(self._datapath)
    for line in open(corpus_path):
      clean_line = self.remove_exclusions(line)
      preprocessed = utils.simple_preprocess(clean_line)
      yield preprocessed
      
      
if __name__ == '__main__':
  # /Lummetry.AI Dropbox/DATA/_allan_data/_indaco/File_1'
  pass
  