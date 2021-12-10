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
        
        in_shape = draw(st.lists(st.integers(min_value=1, max_value=5), min_size = 1, max_size=4))
        # top_k only supports fp32
        in_dtype = draw(st.sampled_from([np.float32]))

        def generate_X_data():
            return np.random.normal(0.0, 5.0, in_shape).astype(in_dtype)

        k_data = draw(st.integers(min_value=1, max_value=4))

        assume(k_data <= in_shape[-1])

        top_k_op = OpConfig(
            type = "top_k",
            inputs = {"X" : ["X_data"]},
            outputs = {"Out": ["Out_data"],
                      "Indices": ["Indices_data"]},
            attrs = {"k" : k_data})
        program_config = ProgramConfig(
            ops=[top_k_op],
            weights={},
            inputs={
                "X_data": TensorConfig(data_gen=partial(generate_X_data)),
            },
            outputs= ["Out_data","Indices_data"])
        
        return program_config


    def sample_predictor_configs(self):
        return self.get_predictor_configs(), [""], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)

if __name__ == "__main__":
    unittest.main(argv=[''])
