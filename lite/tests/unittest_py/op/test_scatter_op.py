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
from functools import partial
import numpy as np
import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
import argparse


class TestScatterOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
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
                    min_value=1, max_value=7), min_size=2, max_size=6))
        update_shape = in_shape
        assume(
            len(update_shape) == len(in_shape) and
            update_shape[1:] == in_shape[1:])

        index_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=len(update_shape)),
                min_size=1,
                max_size=1))
        index_shape[0] = in_shape[0]
        assume(
            len(index_shape) == 1 or
            (len(index_shape) == 2 and index_shape[1] == 1))

        index_type = draw(st.sampled_from(["int32", "int64"]))
        overwrite = draw(st.booleans())

        def generate_data(*args, **kwargs):
            low, high = -10, 10
            dtype = "float32"
            shape = kwargs["shape"]
            if "low" in kwargs:
                low = kwargs["low"]
            if "high" in kwargs:
                high = kwargs["high"]
            if "dtype" in kwargs:
                dtype = kwargs["dtype"]

            if dtype == "int32":
                if low == high:
                    return low * np.ones(shape).astype(np.int32)
                else:
                    return np.random.randint(low, high, shape).astype(np.int32)
            elif dtype == "int64":
                if low == high:
                    return low * np.ones(shape).astype(np.int64)
                else:
                    return np.random.randint(low, high, shape).astype(np.int64)
            elif dtype == "float32":
                return (high - low
                        ) * np.random.random(shape).astype(np.float32) + low

        def generate_index(*args, **kwargs):
            index_np = np.ones(index_shape).astype(np.int64)
            for i in range(index_shape[0]):
                index_np[i] = i
            if kwargs["dtype"] == "int32":
                index_np = index_np.astype(np.int32)
            return index_np

        scatter_op = OpConfig(
            type="scatter",
            inputs={
                "X": ["input_data"],
                "Ids": ["index"],
                "Updates": ["updates"]
            },
            outputs={"Out": ["output_data"]},
            attrs={"overwrite": overwrite})

        program_config = ProgramConfig(
            ops=[scatter_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(
                    generate_data, shape=in_shape)),
                "index": TensorConfig(data_gen=partial(
                    generate_index, dtype=index_type)),
                "updates": TensorConfig(data_gen=partial(
                    generate_data, shape=update_shape))
            },
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["scatter"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 25
        if target_str == "ARM":
            # Make sure to generate enough valid cases for ARM
            max_examples = 100
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
