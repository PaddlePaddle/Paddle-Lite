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

# having diff

class TestFcOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        x86_places = [
                     Place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW),
                     Place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW)
                     ]
        self.enable_testing_on_place(places=x86_places)
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
        in_shape = draw(st.lists(st.integers(min_value=4, max_value=5), min_size = 4, max_size=4))
        in_dtype = draw(st.sampled_from([0,1,2]))

        def generate_X_data():
            if (in_dtype == 0):
                return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)
            elif (in_dtype == 1):
                return np.random.randint(1, 500, in_shape).astype(np.int32)
            elif (in_dtype == 2):
                return np.random.randint(1, 500, in_shape).astype(np.int64)
            elif (in_dtype == 3):
                return np.random.randint(-1, 1, in_shape).astype(np.bool)
            elif (in_dtype == 4):
                return np.random.randint(0, 128, in_shape).astype(np.int8)

        repeat_times_data = draw(st.lists(st.integers(min_value=1, max_value=5), min_size = 1, max_size=4))

        choose_repeat = draw(st.sampled_from(["RepeatTimes", "repeat_times_tensor", "repeat_times"]))

        inputs = {"X" : ["X_data"]}

        def generate_RepeatTimes_data():
            if(choose_repeat == "RepeatTimes"):
                inputs["RepeatTimes"] = ["RepeatTimes_data"]
                return np.array(repeat_times_data).astype(np.int32)
            else:
                return np.random.randint(1, 5, []).astype(np.int32)
        
        # if add repeat_times_tensor, diff arrives.
        def generate_repeat_times_tensor_data():
            if(choose_repeat == "repeat_times_tensor" and False):
                inputs["repeat_times_tensor"] = ["repeat_times_tensor_data"]
                return np.array(repeat_times_data).astype(np.int32)
            else:
                return np.random.randint(1, 5, []).astype(np.int32)

        tile_op = OpConfig(
            type = "tile",
            inputs = inputs,
            outputs = {"Out": ["Out_data"]},
            attrs = {"repeat_times" : repeat_times_data})
        program_config = ProgramConfig(
            ops=[tile_op],
            weights={},
            inputs={
                "X_data": TensorConfig(data_gen=partial(generate_X_data)),
                "RepeatTimes_data": TensorConfig(data_gen=partial(generate_RepeatTimes_data)),
                "repeat_times_tensor_data": TensorConfig(data_gen=partial(generate_repeat_times_tensor_data)),
            },
            outputs= ["Out_data"])
        
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["tile"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)

if __name__ == "__main__":
    unittest.main(argv=[''])
