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
from functools import partial
import random
import numpy as np


class TestUniqueWithCountsOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        host_places = [
            Place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=host_places, thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=100),
                min_size=1,
                max_size=1))
        in_dtype = draw(st.sampled_from([np.float32, np.int32, np.int64]))

        def generate_X_data():
            return np.random.normal(0.0, 5.0, in_shape).astype(in_dtype)

        def generate_IndexTensor():
            return np.random.randint(1, 5, size=in_shape).astype(np.int32)

        unique_with_counts_op = OpConfig(
            type="unique_with_counts",
            inputs={"X": ["input_data"]},
            outputs={
                "Out": ["Out_data"],
                "Index": ["Index_data"],
                "Count": ["Count_data"]
            },
            attrs={"dtype": 2})  # data type for output index,int32

        unique_with_counts_op.outputs_dtype = {"Out_data": in_dtype}
        unique_with_counts_op.outputs_dtype = {"Index_data": np.int32}
        unique_with_counts_op.outputs_dtype = {"Count_data": np.int32}

        program_config = ProgramConfig(
            ops=[unique_with_counts_op],
            weights={
                "Index_data":
                TensorConfig(data_gen=partial(generate_IndexTensor))
            },
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_X_data)),
            },
            outputs=["Out_data", "Index_data", "Count_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), [""], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
