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


def GetWeightShape0(in_dims, start_axis, stop_axis, in_num_col_dims):
    outer = 1
    out_shape = []
    in_dims_size = len(in_dims)

    for i in range(0, start_axis):
        out_shape.append(in_dims[i])
    for i in range(start_axis, stop_axis + 1):
        outer *= in_dims[i]
    out_shape.append(outer)
    for i in range(stop_axis + 1, in_dims_size):
        out_shape.append(in_dims[i])
    #flatten_to_2d
    in_mat_dims_0 = 1
    in_mat_dims_1 = 1
    for i in range(len(out_shape)):
        if (i < in_num_col_dims):
            in_mat_dims_0 = in_mat_dims_0 * out_shape[i]
        else:
            in_mat_dims_1 = in_mat_dims_1 * out_shape[i]

    return in_mat_dims_1


class TestFlattenFcFusePass(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.ARM, PrecisionType.FP32,
                                     DataLayoutType.NCHW)
        self.enable_testing_on_place(TargetType.X86, PrecisionType.FP32,
                                     DataLayoutType.NCHW)
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
        padding_weights = program_config.ops[1].attrs["padding_weights"]
        if predictor_config.target() == TargetType.ARM:
            if padding_weights:
                return False
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=16), min_size=2, max_size=4))
        padding_weights = draw(st.sampled_from([False, True]))
        # OpenCL dose not support this attribute
        if (self.get_target() == 'OpenCL'):
            padding_weights = False
        start_axis = draw(
            st.integers(
                min_value=0, max_value=len(in_shape) - 1))
        in_num_col_dims = draw(
            st.integers(
                min_value=1, max_value=len(in_shape) - 1))

        assume(in_num_col_dims < start_axis + 1)

        stop_axis = draw(
            st.integers(
                min_value=start_axis, max_value=len(in_shape) - 1))

        in_mat_dims_1 = GetWeightShape0(in_shape, start_axis, stop_axis,
                                        in_num_col_dims)
        weights_1 = draw(st.integers(min_value=1, max_value=32))
        weights_0 = in_mat_dims_1
        weights_0 = in_mat_dims_1 + 4 if padding_weights else in_mat_dims_1
        bias_shape0 = weights_1 - 4 if padding_weights else weights_1
        assume(weights_0 > 0 and bias_shape0 > 0)
        weights_shape = [weights_0, weights_1]
        bias_shape = [bias_shape0]
        flatten_op = OpConfig(
            type='flatten_contiguous_range',
            inputs={"X": ["input_data_x"]},
            outputs={
                "Out": ["flatten_output_data"],
                "XShape": ["xshape_data"]
            },
            attrs={
                "data_format": 'nchw',
                "start_axis": start_axis,
                "stop_axis": stop_axis
            })

        fc_inputs = {}
        program_inputs = {}

        def generate_weights(*args, **kwargs):
            return (
                np.random.random(weights_shape).astype(np.float32) - 0.5) * 2

        def generate_bias(*args, **kwargs):
            return (np.random.random(bias_shape).astype(np.float32) - 0.5) * 2

        with_bias = draw(st.sampled_from(
            [True]))  #pass require with_bias as True

        act_type = ""
        if (with_bias and np.random.random() > 0.5):
            act_type = "relu"
        if (with_bias):
            fc_inputs = {
                "Input": ["flatten_output_data"],
                "W": ["weights_data"],
                "Bias": ["bias_data"]
            }
            program_inputs = {
                "input_data_x": TensorConfig(shape=in_shape),
                "weights_data":
                TensorConfig(data_gen=partial(generate_weights)),
                "bias_data": TensorConfig(data_gen=partial(generate_bias))
            }
        else:
            fc_inputs = {
                "Input": ["flatten_output_data"],
                "W": ["weights_data"]
            }
            program_inputs = {
                "input_data_x": TensorConfig(shape=in_shape),
                "weights_data":
                TensorConfig(data_gen=partial(generate_weights))
            }

        fc_op = OpConfig(
            type='fc',
            inputs=fc_inputs,
            outputs={"Out": ["output_data"]},
            attrs={
                "in_num_col_dims": in_num_col_dims,
                "padding_weights": padding_weights,
                "activation_type": act_type,
                "use_mkldnn": False,
                #int8 parameters
                "use_quantizer": False,
                "Scale_in": float(1),
                "Scale_weights": [float(1)],
                "Scale_out": float(1)
            })

        ops = [flatten_op, fc_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={"xshape_data": TensorConfig(shape=in_shape)},
            inputs=program_inputs,
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ['fc'], (1e-4, 1e-4)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=100,
            passes=["lite_flatten_fc_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
