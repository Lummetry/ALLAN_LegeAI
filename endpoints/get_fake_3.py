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
@author: Lummetry.AI - Laurentiu
@project: 
@description:
"""

from libraries.model_server_v2 import FlaskWorker

class GetFake3Worker(FlaskWorker):

  def __init__(self, **kwargs):
    self._inputs = None
    super(GetFake3Worker, self).__init__(prefix_log='[FAKEW]', **kwargs)
    return

  def _load_model(self):
    return

  def _pre_process(self, inputs):
    if 'INPUT_VALUE' not in inputs.keys():
      raise ValueError('input_value should be defined in inputs')

    s = inputs['INPUT_VALUE']
    self._inputs = s
    return s

  def _predict(self, prep_inputs):
    res = [100,200,300,400, 100*int(self._inputs)]
    return res

  def _post_process(self, pred):
    return {'output_value': pred}
