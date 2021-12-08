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
import numpy as np
from functools import partial
import argparse


class TestScaleOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW, thread=[1,2,4])
        self.enable_testing_on_place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW, thread=[1,2,4])
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
        if predictor_config.target() == TargetType.Host:
            return False
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=6))
        bias = draw(st.floats(min_value=-5, max_value=5))
        bias_after_scale = draw(st.booleans())
        scale = draw(st.floats(min_value=-5, max_value=5))
        input_type = draw(st.sampled_from(["int8", "int32", "int64", "float32"]))
        has_scale_tensor = False

        def generate_input_float32(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        input_dict = {"X" : ["input_data"]}
        if has_scale_tensor:
            input_dict["ScaleTensor"] = "scale_tensor_data"

        scale_op = OpConfig(
            type = "scale",
            inputs = input_dict,
            outputs = {"Out" : ["output_data"]},
            attrs = {"bias" : bias,
                    "bias_after_scale" : bias_after_scale,
                    "scale" : scale})

        if has_scale_tensor:
            program_config = ProgramConfig(
                ops=[scale_op],
                weights={},
                inputs={
                    "input_data" : TensorConfig(data_gen=partial(generate_input_float32)),
                    "scale_tensor_data" : TensorConfig(shape=[1,])
                },
                outputs=["output_data"])
        else:
            program_config = ProgramConfig(
                ops=[scale_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input_float32))
                },
                outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["scale"], (1e-5, 1e-5)    

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)

if __name__ == "__main__":
    unittest.main(argv=[''])
