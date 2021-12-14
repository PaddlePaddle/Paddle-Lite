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
        self.enable_testing_on_place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW, thread=[1,2])
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
        # metal demo
        #   fp32 metal kernel
        metal_places = [Place(TargetType.Metal, PrecisionType.FP32, DataLayoutType.MetalTexture2DArray),
                        Place(TargetType.ARM, PrecisionType.FP32),
                        Place(TargetType.Host, PrecisionType.FP32)]
        self.enable_testing_on_place(places=metal_places)
        #   fp16 metal kernel
        fp16_metal_places = [Place(TargetType.Metal, PrecisionType.FP16, DataLayoutType.MetalTexture2DArray),
                        Place(TargetType.ARM, PrecisionType.FP32),
                        Place(TargetType.Host, PrecisionType.FP32)]
        self.enable_testing_on_place(places=fp16_metal_places)

    def is_program_valid(self, program_config: ProgramConfig , predictor_config: CxxConfig) -> bool:
        in_shape = list(program_config.inputs["input_data"].shape)
        in_data_type = program_config.inputs["input_data"].dtype
        if np.int8 == in_data_type:
            print("int8 as Input data type is not supported.")
            return False
        if predictor_config.precision() == PrecisionType.FP16 and in_data_type != np.float16:
            return False
        elif predictor_config.precision() == PrecisionType.FP32 and in_data_type != np.float32:
           return False
        return True

        if in_data_type != np.float16
            print("int8 as Input data type is not supported.")
            return False

        if "ScaleTensor" in program_config.inputs:
            print("ScaleTensor as Input is not supported on Paddle Lite.")
            return False
        if predictor_config.target() == TargetType.Host:
            return False
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=6))
        bias = draw(st.floats(min_value=-5, max_value=5))
        bias_after_scale = draw(st.booleans())
        scale = draw(st.floats(min_value=-5, max_value=5))
        input_type = draw(st.sampled_from(["int32", "int64", "float32"]))
        has_scale_tensor = False # draw(st.booleans())


        def generate_data(*args, **kwargs):
            low, high = -10, 10
            dtype = "float32"
            shape = kwargs["shape"]
            if "low" in kwargs:
                low = kwargs["low"]
            if "high" in kwargs:
                high = kwargs["high"]
            if "dtype" in kwargs:
                dtype = kwargs["dtype"]

            if dtype == "int32":
                if low == high:
                    return low * np.ones(shape).astype(np.int32)
                else:
                    return np.random.randint(low, high, shape).astype(np.int32)
            elif dtype == "int64":
                if low == high:
                    return low * np.ones(shape).astype(np.int64)
                else:
                    return np.random.randint(low, high, shape).astype(np.int64)
            elif dtype == "float32":
                return high * np.random.random(shape).astype(np.float32) + low

        input_dict = {"X" : ["input_data"]}
        input_data_dict = {"input_data" : TensorConfig(data_gen=partial(generate_data, dtype=input_type, shape=in_shape))}
        if has_scale_tensor:
            input_dict["ScaleTensor"] = "scale_tensor_data"
            input_data_dict["scale_tensor_data"] = TensorConfig(shape=[1,])

        scale_op = OpConfig(
            type = "scale",
            inputs = input_dict,
            outputs = {"Out" : ["output_data"]},
            attrs = {"bias" : bias,
                    "bias_after_scale" : bias_after_scale,
                    "scale" : scale})

        program_config = ProgramConfig(
            ops=[scale_op],
            weights={},
            inputs=input_data_dict,
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
