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


class TestShuffleChannelFusePass(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
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
        #x86
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        #metal
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
        in_shape = program_config.ops[0].attrs["shape"]
        if target_type in [TargetType.Metal]:
            if in_shape[0] != 1:
                return False
        return True

    def sample_program_configs(self, draw):
        reshape1_output_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=32), min_size=5, max_size=5))
        reshape1_input_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=32), min_size=4, max_size=4))
        reshape2_output_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=32), min_size=4, max_size=4))

        reshape1_input_shape[0] = reshape1_output_shape[0]
        reshape1_input_shape[1] = reshape1_output_shape[
            1] * reshape1_output_shape[2]
        reshape1_input_shape[2] = reshape1_output_shape[3]
        reshape1_input_shape[3] = reshape1_output_shape[4]

        reshape2_output_shape[0] = reshape1_input_shape[0]
        reshape2_output_shape[1] = -1
        reshape2_output_shape[2] = reshape1_input_shape[2]
        reshape2_output_shape[3] = reshape1_input_shape[3]

        shuffle_channel_fuser_type = draw(
            st.sampled_from(
                ["shuffle_channel_fuser1", "shuffle_channel_fuser2"]))

        if shuffle_channel_fuser_type == "shuffle_channel_fuser1":
            reshape1_op = OpConfig(
                type="reshape",
                inputs={"X": ["reshape1_input_x"]},
                outputs={"Out": ["reshape1_output"]},
                attrs={"shape": reshape1_output_shape})

            transpose_op = OpConfig(
                type="transpose",
                inputs={"X": ["reshape1_output"]},
                outputs={"Out": ["transpose_output"]},
                attrs={"use_mkldnn": False,
                       "axis": [0, 2, 1, 3, 4]})

            reshape2_op = OpConfig(
                type="reshape",
                inputs={"X": ["transpose_output"]},
                outputs={"Out": ["output_data"]},
                attrs={"shape": reshape2_output_shape})
        else:
            reshape1_op = OpConfig(
                type="reshape2",
                inputs={"X": ["reshape1_input_x"]},
                outputs={
                    "Out": ["reshape1_output"],
                    "XShape": ["reshape1_XShape_data"]
                },
                attrs={"shape": reshape1_output_shape})

            transpose_op = OpConfig(
                type="transpose2",
                inputs={"X": ["reshape1_output"]},
                outputs={
                    "Out": ["transpose_output"],
                    "XShape": ["transpose_XShape_data"]
                },
                attrs={"use_mkldnn": False,
                       "axis": [0, 2, 1, 3, 4]})

            reshape2_op = OpConfig(
                type="reshape2",
                inputs={"X": ["transpose_output"]},
                outputs={
                    "Out": ["output_data"],
                    "XShape": ["reshape2_XShape_data"]
                },
                attrs={"shape": reshape2_output_shape})

        ops = [reshape1_op, transpose_op, reshape2_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={
                "reshape1_XShape_data": TensorConfig(shape=[5]),
                "transpose_XShape_data": TensorConfig(shape=[5]),
                "reshape2_XShape_data": TensorConfig(shape=[4])
            },
            inputs={
                "reshape1_input_x": TensorConfig(shape=reshape1_input_shape)
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        config_lists = self.get_predictor_configs()
        for config in config_lists:
            if config.target() in [TargetType.Metal]:
                atol, rtol = 1e-2, 1e-2
        return self.get_predictor_configs(), ['shuffle_channel'], (atol, rtol)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 25
        if target_str in ["Metal"]:
            # Make sure to generate enough valid cases for specific targets
            max_examples = 1300
        self.run_and_statis(
            quant=False,
            max_examples=max_examples,
            passes=["lite_shuffle_channel_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
