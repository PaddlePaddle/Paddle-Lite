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

from auto_scan_test import FusePassAutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestTransposeSoftmaxTransposeFusePass(FusePassAutoScanTest):
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
        # metal has diff(test case: transpose1_input_shape = [1, 1, 64, 64])
        # metal_places = [
        #     Place(TargetType.Metal, PrecisionType.FP32,
        #           DataLayoutType.MetalTexture2DArray),
        #     Place(TargetType.Metal, PrecisionType.FP16,
        #           DataLayoutType.MetalTexture2DArray),
        #     Place(TargetType.ARM, PrecisionType.FP32),
        #     Place(TargetType.Host, PrecisionType.FP32)
        # ]
        # self.enable_testing_on_place(places=metal_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        dim = draw(st.sampled_from([2, 3, 4]))
        transpose_type = draw(st.sampled_from(["transpose", "transpose2"]))

        #default dim = 4
        transpose1_input_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=64), min_size=4, max_size=4))
        transpose1_axis = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=3), min_size=4, max_size=4))
        assume(sorted(transpose1_axis) == [0, 1, 2, 3])
        transpose2_axis = [
            transpose1_axis.index(0), transpose1_axis.index(1),
            transpose1_axis.index(2), transpose1_axis.index(3)
        ]

        if dim == 2:
            transpose1_input_shape = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=64),
                    min_size=2,
                    max_size=2))
            transpose1_axis = draw(
                st.lists(
                    st.integers(
                        min_value=0, max_value=1),
                    min_size=2,
                    max_size=2))
            assume(sorted(transpose1_axis) == [0, 1])
            transpose2_axis = [
                transpose1_axis.index(0), transpose1_axis.index(1)
            ]
        elif dim == 3:
            transpose1_input_shape = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=64),
                    min_size=3,
                    max_size=3))
            transpose1_axis = draw(
                st.lists(
                    st.integers(
                        min_value=0, max_value=2),
                    min_size=3,
                    max_size=3))
            assume(sorted(transpose1_axis) == [0, 1, 2])
            transpose2_axis = [
                transpose1_axis.index(0), transpose1_axis.index(1),
                transpose1_axis.index(2)
            ]

        if transpose_type == "transpose":
            transpose1_op = OpConfig(
                type="transpose",
                inputs={"X": ["transpose1_input_x"]},
                outputs={"Out": ["transpose1_output"]},
                attrs={"axis": transpose1_axis})

            softmax_op = OpConfig(
                type="softmax",
                inputs={"X": ["transpose1_output"]},
                outputs={"Out": ["softmax_output"]},
                attrs={"axis": -1})

            transpose2_op = OpConfig(
                type="transpose",
                inputs={"X": ["softmax_output"]},
                outputs={"Out": ["output_data"]},
                attrs={"axis": transpose2_axis})
        elif transpose_type == "transpose2":
            transpose1_op = OpConfig(
                type="transpose2",
                inputs={"X": ["transpose1_input_x"]},
                outputs={
                    "Out": ["transpose1_output"],
                    "XShape": ["transpose1_XShape"]
                },
                attrs={"axis": transpose1_axis})

            softmax_op = OpConfig(
                type="softmax",
                inputs={"X": ["transpose1_output"]},
                outputs={"Out": ["softmax_output"]},
                attrs={"axis": -1})

            transpose2_op = OpConfig(
                type="transpose2",
                inputs={"X": ["softmax_output"]},
                outputs={
                    "Out": ["output_data"],
                    "XShape": ["transpose2_XShape"]
                },
                attrs={"axis": transpose2_axis})

        ops = [transpose1_op, softmax_op, transpose2_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "transpose1_input_x":
                TensorConfig(shape=transpose1_input_shape)
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ['softmax'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=50,
            passes=["lite_transpose_softmax_transpose_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
