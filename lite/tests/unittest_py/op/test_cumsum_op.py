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


class TestCumsumOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.Any,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        if program_config.ops[0].attrs['reverse'] == True:
            return False
        return True

    def sample_program_configs(self, draw):
        input_data_x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=1, max_size=8))
        x_dims_size = len(input_data_x_shape)
        flatten = draw(st.booleans())
        axis = draw(
            st.integers(
                min_value=-x_dims_size, max_value=x_dims_size - 1))
        if flatten:
            axis = -1
        exclusive = draw(st.booleans())
        reverse = draw(st.booleans())

        cumsum_op = OpConfig(
            type="cumsum",
            inputs={"X": ["input_data_x"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "axis": axis,
                "flatten": flatten,
                "exclusive": exclusive,
                "reverse": reverse
            })
        program_config = ProgramConfig(
            ops=[cumsum_op],
            weights={},
            inputs={"input_data_x": TensorConfig(shape=input_data_x_shape), },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["cumsum"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def reverse_attr_not_support_teller(program_config, predictor_config):
            if program_config.ops[0].attrs['reverse'] == True:
                return False
            return True

        self.add_ignore_check_case(
            reverse_attr_not_support_teller,
            IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Cumsum op current isn't support the attr['reverse'] == True!")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
