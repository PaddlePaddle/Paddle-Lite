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
import sys
sys.path.append('..')

from auto_scan_test import FusePassAutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
from test_elementwise_util import trim_trailing_singular_dims, check_input_shape_available
import hypothesis.strategies as st


class TestElementwiseScaleFuse(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape_x = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=20), min_size=2, max_size=5))
        in_shape_x = draw(st.sampled_from([in_shape_x, []]))
        axis = -1
        sigmoid_op = OpConfig(
            type='sigmoid',
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"]},
            attrs={})
        elementwise_op = OpConfig(
            type='elementwise_mul',
            inputs={"X": ["output_data"],
                    "Y": ["input_data"]},
            outputs={"Out": ["elementwise_output_data"]},
            attrs={"axis": axis})

        ops = [sigmoid_op, elementwise_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={"input_data": TensorConfig(shape=in_shape_x)},
            outputs=["elementwise_output_data"])
        return program_config

    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), ['swish'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 100
        self.run_and_statis(
            quant=False,
            max_examples=max_examples,
            passes=["lite_sigmoid_elementmul_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
