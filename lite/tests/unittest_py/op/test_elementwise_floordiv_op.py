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
sys.path.append('../')

from auto_scan_test import AutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st
import numpy as np
from functools import partial

class TestElementwiseFloorDivOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW, thread=[1,4])
        self.enable_testing_on_place(TargetType.ARM, [PrecisionType.FP32, PrecisionType.INT32, PrecisionType.INT64], DataLayoutType.NCHW, thread=[1,4])

    def is_program_valid(self, program_config: ProgramConfig , predictor_config: CxxConfig) -> bool:
        if program_config.inputs['input_data_x'].dtype == np.float:
            return False
        return True

    def sample_program_configs(self, draw):
        input_data_x_shape = draw(st.lists(st.integers(min_value=1, max_value=20), min_size = 4, max_size = 4))
        axis = draw(st.integers(min_value=0, max_value=(len(input_data_x_shape) - 1)))
        input_data_y_shape = input_data_x_shape[axis:] 
        input_data_type = draw(st.sampled_from([np.int32, np.int64, np.float]))

        def gen_input_data(*args, **kwargs):
            return np.random.randint(1, 20, size=(kwargs['shape'])).astype(kwargs['dtype'])

        elementwise_floordiv_op = OpConfig(
            type = "elementwise_floordiv",
            inputs = {"X" : ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs = {"Out": ["output_data"]},
            attrs = {"axis" : axis})
        program_config = ProgramConfig(
            ops=[elementwise_floordiv_op],
            weights={},
            inputs={
                "input_data_x": TensorConfig(data_gen=partial(gen_input_data, shape=input_data_x_shape, dtype=input_data_type)),
                "input_data_y": TensorConfig(data_gen=partial(gen_input_data, shape=input_data_y_shape, dtype=input_data_type))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), ["elementwise_floordiv"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)

if __name__ == "__main__":
    unittest.main(argv=[''])
