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

from auto_scan_test import FusePassAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestInstanceNormActivationFusePass(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        #opencl
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

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=64), min_size=4, max_size=4))
        epsilon_data = draw(st.floats(min_value=0.0, max_value=0.001))

        act_type = draw(st.sampled_from(
            ['relu']))  #opencl instance_norm kernel only support relu

        instance_norm_op = OpConfig(
            type="instance_norm",
            inputs={
                "X": ["input_data"],
                "Scale": ["scale_data"],
                "Bias": ["bisa_data"]
            },
            outputs={
                "Y": ["y_output_data"],
                "SavedMean": ["SavedMean_data"],
                "SavedVariance": ["SavedVariance_data"]
            },
            attrs={"epsilon": epsilon_data})

        active_op = OpConfig(
            type=act_type,
            inputs={"X": ["y_output_data"]},
            outputs={"Out": ["output_data"]},
            attrs={})

        ops = [instance_norm_op, active_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data": TensorConfig(shape=in_shape),
                "scale_data": TensorConfig(shape=[in_shape[1]]),
                "bisa_data": TensorConfig(shape=[in_shape[1]])
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ['instance_norm'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["lite_instance_norm_activation_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
