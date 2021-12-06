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

from auto_scan_base import AutoScanBaseTest, IgnoreReasonsBase
import numpy as np
import abc
import enum
import unittest
from typing import Optional, List, Callable, Dict, Any, Set
import os
import paddle
import rpyc
import copy
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True


class AutoScanTest(AutoScanBaseTest):
    def run_lite_config(self, model, params, feed_data, pred_config) -> Dict[str, np.ndarray]:
        conn = rpyc.connect("localhost", 18812)
        out, model = conn.root.run_lite_model(model,params,feed_data, pred_config)
        result_res = copy.deepcopy(out)
        return result_res, model

class FusePassAutoScanTest(AutoScanTest):
    def run_and_statis(
            self,
            quant=False,
            max_examples=100,
            reproduce=None,
            min_success_num=25,
            max_duration=180,
            passes=None ):
        assert passes is not None, "Parameter of passes must be defined in function run_and_statis."
        super().run_and_statis(quant, max_examples, reproduce, min_success_num, max_duration, passes)
