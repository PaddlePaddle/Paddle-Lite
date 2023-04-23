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


class TestEmptyOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host, [PrecisionType.Any],
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=1, max_size=6))
        in_shape = draw(st.sampled_from([in_shape, []]))
        tensor_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=1, max_size=4))
        dtype = draw(st.sampled_from([2, 3, 5]))
        with_shape = draw(st.sampled_from([True, False]))
        with_tensor = draw(st.sampled_from([True, False]))

        def generate_shape_tensor(*args, **kwargs):
            return np.array(tensor_shape).astype(np.int32)

        if with_shape and with_tensor:
            empty_op = OpConfig(
                type="empty",
                inputs={"ShapeTensor": ["shape_tensor"]},
                outputs={"Out": ["output_data"]},
                attrs={"dtype": dtype,
                       "shape": in_shape})
        elif with_tensor:
            empty_op = OpConfig(
                type="empty",
                inputs={"ShapeTensor": ["shape_tensor"]},
                outputs={"Out": ["output_data"]},
                attrs={"dtype": dtype})
        elif with_shape:
            empty_op = OpConfig(
                type="empty",
                inputs={},
                outputs={"Out": ["output_data"]},
                attrs={"dtype": dtype,
                       "shape": in_shape})
        else:
            empty_op = OpConfig(
                type="empty",
                inputs={},
                outputs={"Out": ["output_data"]},
                attrs={"dtype": dtype})
        if dtype == 2:
            empty_op.outputs_dtype = {"output_data": np.int32}
        elif dtype == 3:
            empty_op.outputs_dtype = {"output_data": np.int64}
        else:
            empty_op.outputs_dtype = {"output_data": np.float32}

        program_config = ProgramConfig(
            ops=[empty_op],
            weights={},
            inputs={
                "shape_tensor":
                TensorConfig(data_gen=partial(generate_shape_tensor))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["empty"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
