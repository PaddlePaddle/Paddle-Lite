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
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st
import numpy as np
from functools import partial


def check_broadcast(x_shape, y_shape, axis):
    if x_shape == y_shape:
        return True
    else:
        x_dims_size = len(x_shape)
        y_dims_size = len(y_shape)
        max_dims_size = max(x_dims_size, y_dims_size)
        if (x_dims_size == y_dims_size):
            # The axis should be -1 or 0 while the dimension of tensor X is equal to the dimension of tensor Y.
            if not (axis == -1 or axis == 0):
                return False
        # The axis should be met [-max(x_dims_size, y_dims_size), max(x_dims_size, y_dims_size))
        if not (axis >= (-max_dims_size) and axis < max_dims_size):
            return False
        if axis < 0:
            axis = abs(x_dims_size - y_dims_size) + axis + 1
        if not (axis >= 0 and axis < max_dims_size):
            return False
        # The axis should be met axis + min(x_dims_size, y_dims_size) <= max(x_dims_size, y_dims_size)
        if axis + min(x_dims_size, y_dims_size) > max_dims_size:
            return False
        x_dims_array = [1 for i in range(max_dims_size)]
        y_dims_array = [1 for i in range(max_dims_size)]
        if x_dims_size > y_dims_size:
            x_dims_array = x_shape
            for i in range(y_dims_size):
                y_dims_array[axis + i] = y_shape[i]
        else:
            y_dims_array = y_shape
            for i in range(x_dims_size):
                x_dims_array[axis + i] = x_shape[i]
        for i in range(max_dims_size):
            broadcast_flag = (x_dims_array[i] == y_dims_array[i]) or (
                x_dims_array[i] <= 1) or (y_dims_array[i] <= 1)
            if not broadcast_flag:
                return False
        return True


class TestBitwiseOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        host_places = [
            Place(TargetType.Host, PrecisionType.Any, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=host_places)

    def sample_program_configs(self, draw):
        input_type = draw(st.sampled_from(["int32", "int64"]))
        input_data_x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=1, max_size=4))
        input_data_y_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=1, max_size=4))
        axis = -1  # paddle2.4.2 only support -1
        input_data_x_shape = draw(st.sampled_from([input_data_x_shape, []]))
        input_data_y_shape = draw(st.sampled_from([input_data_y_shape, []]))
        assume(
            check_broadcast(input_data_x_shape, input_data_y_shape, axis) ==
            True)
        if axis < 0:
            axis = abs(len(input_data_x_shape) - len(
                input_data_y_shape)) + axis + 1

        def generate_input(*args, **kwargs):
            if kwargs["type"] == "bool":
                return np.random.choice([True, False],
                                        kwargs["shape"]).astype(bool)
            if kwargs["type"] == "uint8":
                return np.random.normal(0, 255,
                                        kwargs["shape"]).astype(np.uint8)
            if kwargs["type"] == "int8":
                return np.random.normal(-128, 127,
                                        kwargs["shape"]).astype(np.int8)
            if kwargs["type"] == "int16":
                return np.random.normal(-32768, 32767,
                                        kwargs["shape"]).astype(np.int16)
            if kwargs["type"] == "int32":
                return np.random.normal(-2147483648, 2147483647,
                                        kwargs["shape"]).astype(np.int32)
            if kwargs["type"] == "int64":
                return np.random.normal(-9223372036854775808,
                                        9223372036854775807,
                                        kwargs["shape"]).astype(np.int64)

        bitwise_op = OpConfig(
            type="bitwise_or",
            inputs={
                "X": ["input_data"],
                "Y": ["input_data1"],
            },
            outputs={"Out": ["output_data"]},
            attrs={"axis": axis})
        if (input_type == "int32"):
            bitwise_op.outputs_dtype = {"output_data": np.int32}
        elif (input_type == "int64"):
            bitwise_op.outputs_dtype = {"output_data": np.int64}

        program_config = ProgramConfig(
            ops=[bitwise_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(
                    generate_input, type=input_type,
                    shape=input_data_x_shape)),
                "input_data1": TensorConfig(data_gen=partial(
                    generate_input, type=input_type,
                    shape=input_data_y_shape)),
            },
            outputs=["output_data"])

        return program_config

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["bitwise_or"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=500)


if __name__ == "__main__":
    unittest.main(argv=[''])
