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
from functools import partial
import random
import numpy as np


class TestBatchNormOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.FP16,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.OpenCL, PrecisionType.Any,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.Any,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=opencl_places)
        metal_places = [
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.Metal, PrecisionType.FP16,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.ARM, PrecisionType.FP32),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=metal_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=32), min_size=4, max_size=4))
        is_test_val = draw(st.sampled_from([True, False]))
        epsilon = draw(st.floats(min_value=0.00001, max_value=0.001))
        momentum = draw(st.floats(min_value=0.1, max_value=0.9))

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_scale(*args, **kwargs):
            return np.random.random([in_shape[1]]).astype(np.float32) + 0.5

        def generate_bias(*args, **kwargs):
            return np.random.random([in_shape[1]]).astype(np.float32)

        def generate_mean(*args, **kwargs):
            return np.random.random([in_shape[1]]).astype(np.float32)

        def generate_variance(*args, **kwargs):
            return np.random.random([in_shape[1]]).astype(np.float32)

        outputs = [
            "output_data", "mean_data", "variance_data", "saved_mean",
            "saved_variance"
        ]
        if self.get_target() == "Metal":
            outputs = ["output_data"]

        batch_norm_ops = OpConfig(
            type="batch_norm",
            inputs={
                "X": ["input_data"],
                "Scale": ["scale_data"],
                "Bias": ["bias_data"],
                "Mean": ["mean_data"],
                "Variance": ["variance_data"]
            },
            outputs={
                "Y": ["output_data"],
                "MeanOut": ["mean_data"],
                "VarianceOut": ["variance_data"],
                "SavedMean": ["saved_mean"],
                "SavedVariance": ["saved_variance"]
            },
            attrs={
                "is_test": False,
                "trainable_statistics": False,
                "data_layout": "NCHW",
                "use_global_stats": True,
                "epsilon": epsilon,
                "momentum": momentum
            })
        program_config = ProgramConfig(
            ops=[batch_norm_ops],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input)),
                "scale_data": TensorConfig(data_gen=partial(generate_scale)),
                "bias_data": TensorConfig(data_gen=partial(generate_bias)),
                "mean_data": TensorConfig(data_gen=partial(generate_mean)),
                "variance_data":
                TensorConfig(data_gen=partial(generate_variance)),
            },
            outputs=outputs)
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        if target_str == "Metal":
            atol, rtol = 8e-3, 8e-3
        return self.get_predictor_configs(), ["batch_norm"], (atol, rtol)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            x_shape = list(program_config.inputs["input_data"].shape)
            if predictor_config.target() == TargetType.Metal:
                if x_shape[0] != 1:
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.ACCURACY_ERROR,
            "The op output has diff in a specific case. We need to fix it as soon as possible."
        )

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 25
        if target_str == "Metal":
            # Make sure to generate enough valid cases for Metal
            max_examples = 1500
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
