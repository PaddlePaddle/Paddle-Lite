# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from auto_scan_base import AutoScanBaseTest, SkipReasonsBase
import numpy as np
import abc
import enum
import unittest
from typing import Optional, List, Callable, Dict, Any, Set
from paddlelite.lite import *

SkipReasons = SkipReasonsBase

class AutoScanTest(AutoScanBaseTest):
    def run_lite_config(self, model, params, inputs, pred_config) -> Dict[str, np.ndarray]:
        valid_places=[Place(TargetType.Host, PrecisionType.FP32)]
        config = CxxConfig()
        config.set_valid_places(valid_places)
        config.set_model_buffer(model, len(model), params, len(params))
        predictor = create_paddle_predictor(config)
        for name in inputs:
            input_tensor = predictor.get_input_by_name(name)
            input_tensor.from_numpy(inputs[name]['data'])
            if inputs[name]['lod'] is not None:
                input_tensor.set_lod(inputs[name]['lod'])
        predictor.run()
        result = {}
        for out_name in predictor.get_output_names():
            result[out_name] = predictor.get_output_by_name(out_name).numpy()
        return result
