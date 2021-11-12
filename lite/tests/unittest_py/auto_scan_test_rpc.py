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
import os

import rpyc
rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True


SkipReasons = SkipReasonsBase

class AutoScanTest(AutoScanBaseTest):
    def run_lite_config(self, model, params, feed_data, pred_config) -> Dict[str, np.ndarray]:
        conn = rpyc.connect("localhost", 18812, config = rpyc.core.protocol.DEFAULT_CONFIG)
        out, model = conn.root.run_lite_model(model,params,feed_data, pred_config)
        return out, model

class FusePassAutoScanTest(AutoScanBaseTest):
    def assert_op_size(self, fusion_before_num, fusion_after_num, origin_model):
        model_bytes = paddle.static.load_from_file(self.cache_dir+"/model")
        pg = paddle.static.deserialize_program(model_bytes)
        main_block = pg.desc.block(0)
        after_op_size = main_block.op_size()
        pg = paddle.static.deserialize_program(origin_model)
        main_block = pg.desc.block(0)
        before_op_size = main_block.op_size()
        self.assertTrue(before_op_size == fusion_before_num,
                        'before fusion op size is {}, but got {}!'.format(
                            before_op_size, fusion_before_num))
        self.assertTrue(after_op_size == fusion_after_num,
                        'after fusion op size is {}, but got {}!'.format(
                            after_op_size, fusion_after_num))

    def run_lite_config(self, model, params, inputs, pred_config) -> Dict[str, np.ndarray]:
        conn = rpyc.connect("localhost", 18812, config = rpyc.core.protocol.DEFAULT_CONFIG)
        out, model = conn.root.run_lite_model(model,params,feed_data, pred_config)
        return out, model