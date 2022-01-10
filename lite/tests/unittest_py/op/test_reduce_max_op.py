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


class TestReduceMaxOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 2])
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
        target_type = predictor_config.target()
        in_shape = list(program_config.inputs["input_data"].shape)
        axis = program_config.ops[0].attrs["dim"]
        keep_dim = program_config.ops[0].attrs["keep_dim"]
        if target_type == TargetType.OpenCL:
            if keep_dim == False:
                return False
        if target_type == TargetType.Metal:
            if keep_dim == False or axis[0] != 1 or in_shape[0] != 1:
                return False
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=4, max_size=4))
        keep_dim = draw(st.booleans())
        axis = draw(st.integers(min_value=-1, max_value=3))
        assume(axis < len(in_shape))

        if isinstance(axis, int):
            axis = [axis]
        reduce_all_data = True if axis == None or axis == [] else False

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        build_ops = OpConfig(
            type="reduce_max",
            inputs={"X": ["input_data"], },
            outputs={"Out": ["output_data"], },
            attrs={
                "dim": axis,
                "keep_dim": keep_dim,
                "reduce_all": reduce_all_data,
            })
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input)),
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        if target_str == "Metal":
            atol, rtol = 2e-4, 2e-4

        return self.get_predictor_configs(), ["reduce_max"], (atol, rtol)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 100
        if target_str == "Metal":
            # Make sure to generate enough valid cases for Metal
            max_examples = 3000
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
