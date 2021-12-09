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
import argparse
import numpy as np
from functools import partial

class TestCastOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW, thread=[1, 4])
        self.enable_testing_on_place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW, thread=[1, 4])
        self.enable_testing_on_place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW, thread=[1, 4])
        opencl_places = [Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageDefault),
                          Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageFolder),
                          Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageDefault),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageFolder),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
                          Place(TargetType.Host, PrecisionType.FP32)]
        self.enable_testing_on_place(places=opencl_places)

    def is_program_valid(self, program_config: ProgramConfig , predictor_config: CxxConfig) -> bool:
        if program_config.ops[0].attrs["in_dtype"] == 0 or program_config.ops[0].attrs["out_dtype"] == 0:
            return False
        else:
            return True

    def sample_program_configs(self, draw):
        in_shape = draw(st.lists(st.integers(min_value=1, max_value=10), min_size=1, max_size=4))
        # BOOL = 0;INT16 = 1;INT32 = 2;INT64 = 3;FP16 = 4;FP32 = 5;FP64 = 6;
        in_dtype = draw(st.sampled_from([0, 2, 3, 5]))
        out_dtype = draw(st.sampled_from([0, 2, 3, 5]))
        # BOOL and INT16 and FP16 and FP64 paddle-lite doesn't support
        def generate_input(*args, **kwargs):
            if in_dtype == 0:
                return np.random.random(in_shape).astype(np.bool)
            elif in_dtype == 1:
                return np.random.random(in_shape).astype(np.int16)
            elif in_dtype == 2:
                return np.random.random(in_shape).astype(np.int32)
            elif in_dtype == 3:
                return np.random.random(in_shape).astype(np.int64)
            elif in_dtype == 4:
                return np.random.random(in_shape).astype(np.float16)
            elif in_dtype == 5:
                return np.random.random(in_shape).astype(np.float32)
            else:
                print("in_dtype is error! ", in_dtype)
        cast_op = OpConfig(
            type = "cast",
            inputs = {"X" : ["input_data"]},
            outputs = {"Out" : ["output_data"]},
            attrs = {"in_dtype" : in_dtype,
                    "out_dtype" : out_dtype})

        program_config = ProgramConfig(
            ops=[cast_op],
            weights={},
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["cast"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=250)

if __name__ == "__main__":
    unittest.main(argv=[''])
