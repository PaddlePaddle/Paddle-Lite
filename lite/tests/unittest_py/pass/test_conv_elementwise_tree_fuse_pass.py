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
sys.path.append('.')

from auto_scan_test import FusePassAutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
from test_conv_util import UpdatePaddingAndDilation, ConvOutputSize, ConvTransposeOutputSize
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestConvElementwiseTreeFuse(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
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

        #conv param or conv_transpose param
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=128),
                min_size=3,
                max_size=3))
        in_shape = [draw(st.integers(min_value=1, max_value=4))] + in_shape
        weight_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=2, max_size=2))
        weight_shape = weight_shape + [1, 1]
        paddings = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=2), min_size=2, max_size=2))
        dilations = draw(st.sampled_from([[1, 1], [2, 2]]))
        groups = draw(st.sampled_from([1, 2, in_shape[1]]))
        padding_algorithm = draw(st.sampled_from(["VALID", "SAME"]))
        strides = draw(st.sampled_from([[1, 1], [2, 2]]))
        output_padding = draw(
            st.sampled_from([[], draw(
                st.lists(
                    st.integers(
                        min_value=0, max_value=16),
                    min_size=2,
                    max_size=2))]))
        scale_in = draw(st.floats(min_value=0.001, max_value=0.1))
        scale_out = draw(st.floats(min_value=0.001, max_value=0.1))

        conv_out_shape = []
        paddings_, dilations_ = UpdatePaddingAndDilation(
            in_shape, weight_shape, paddings, dilations, groups,
            padding_algorithm, strides)

        self.depthwise = in_shape[1] == weight_shape[1] and in_shape[
            1] == groups
        assume(in_shape[1] == weight_shape[1] * groups)
        assume(weight_shape[0] % groups == 0)
        conv_out_shape = [in_shape[0], weight_shape[0]]
        oh, ow = ConvOutputSize(in_shape, weight_shape, dilations_, paddings_,
                                strides)
        conv_out_shape = conv_out_shape + [oh, ow]
        assume(oh > 0 and ow > 0)

        conv_type = "conv2d"
        conv_attrs = {
            "data_format": 'nchw',
            "dilations": dilations,
            "padding_algorithm": padding_algorithm,
            "groups": groups,
            "paddings": paddings,
            "strides": strides,
            "Scale_in": scale_in,
            "Scale_out": scale_out
        }

        conv_op = OpConfig(
            type=conv_type,
            inputs={"Input": ["input_data"],
                    "Filter": ["filter_data"]},
            outputs={"Output": ["conv_output_data"]},
            attrs=conv_attrs)

        elementwise_add_op = OpConfig(
            type="elementwise_add",
            inputs={"X": ["add_input_data"],
                    "Y": ["conv_output_data"]},
            outputs={"Out": ["output_data"]},
            attrs={"axis": -1})

        ops = [conv_op, elementwise_add_op]
        self.ops = ops
        program_config = ProgramConfig(
            ops=ops,
            weights={"filter_data": TensorConfig(shape=weight_shape)},
            inputs={
                "input_data": TensorConfig(shape=in_shape),
                "add_input_data": TensorConfig(shape=conv_out_shape)
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), [self.ops[0].type], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=100,
            passes=["lite_conv_elementwise_tree_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
