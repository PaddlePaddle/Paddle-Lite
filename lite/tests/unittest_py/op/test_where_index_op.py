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
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
from functools import partial
import random
import numpy as np


class TestFcOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        x86_places = [
                     Place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW),
                     Place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW)
                     ]
        self.enable_testing_on_place(places=x86_places)
        
        arm_places = [
                     Place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW),
                     Place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW)
                     ]
        self.enable_testing_on_place(places=arm_places)
        
        # opencl demo
        opencl_places = [Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageDefault),
                          Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageFolder),
                          Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageDefault),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageFolder),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
                          Place(TargetType.Host, PrecisionType.FP32)    
                        ]
        self.enable_testing_on_place(places=opencl_places)

    def is_program_valid(self, program_config: ProgramConfig , predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(st.lists(st.integers(min_value=1, max_value=10), min_size = 4, max_size=4))
        in_dtype = draw(st.sampled_from([0, 1, 2, 3]))

        def generate_Condition_data():
            if (in_dtype == 0):
                return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)
            elif (in_dtype == 1):
                return np.random.randint(1, 500, in_shape).astype(np.int32)
            elif (in_dtype == 2):
                return np.random.randint(1, 500, in_shape).astype(np.int64)
            elif (in_dtype == 3):
                return np.random.choice([True,False], in_shape , replace=True)
        
        where_index_op = OpConfig(
            type = "where_index",
            inputs = {"Condition" : ["Condition_data"]},
            outputs = {"Out": ["Out_data"]},
            attrs = {})
        where_index_op.outputs_dtype = {"Out_data": np.int64}
        
        program_config = ProgramConfig(
            ops=[where_index_op],
            weights={},
            inputs={
                "Condition_data": TensorConfig(data_gen=partial(generate_Condition_data))
            },
            outputs= ["Out_data"])
        
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), [""], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)

if __name__ == "__main__":
    unittest.main(argv=[''])
