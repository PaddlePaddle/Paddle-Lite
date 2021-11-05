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

from auto_scan_test import AutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
from paddlelite.lite import *

class TestAssignOp(AutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self, *args, **kwargs):
        def generate_input(*args, **kwargs):
            return np.random.random(kwargs['in_shape']).astype(np.float32)

        ops_config = [{
            "op_type": "assign",
            "op_inputs": {
                "X": ["input_data"],
            },
            "op_outputs": {
                "Out": ["output_data"]
            },
            "op_attrs": {
            }
        }]

        ops = self.generate_op_config(ops_config)

        program_config = ProgramConfig(
            ops=ops,
            weights={
            },
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input, *args, **kwargs)),
            },
            outputs=["output_data"])

        yield program_config

    def sample_predictor_configs(self, program_config):
        config = self.create_config(valid_places=[Place(TargetType.Host, PrecisionType.FP32)])
        yield config, (1e-5, 1e-5)

    def add_skip_case(self):
        pass

    @given(
        in_shape=st.lists(
            st.integers(
                min_value=1, max_value=8)))
    def test(self, *args, **kwargs):
        self.add_skip_pass_case()
        self.run_test(quant=False, *args, **kwargs)


if __name__ == "__main__":
    unittest.main()
