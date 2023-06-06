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


class TestScatterNdAddOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        in_dtype = program_config.inputs["input_data"].dtype
        index_dtype = program_config.inputs["index"].dtype
        if in_dtype == "float32" and index_dtype == "int32":
            return True
        else:
            return False

    def sample_program_configs(self, draw):
        def judge_update_shape(ref_shape, index_shape):
            update_shape = []
            for i in range(len(index_shape) - 1):
                update_shape.append(index_shape[i])
            for i in range(index_shape[-1], len(ref_shape), 1):
                update_shape.append(ref_shape[i])
            return update_shape

        input_type = draw(st.sampled_from(["int32", "int64", "float32"]))
        index_type = draw(st.sampled_from(["int32", "int64"]))
        out_dtype_dict = {
            "int32": np.int32,
            "int64": np.int64,
            "float32": np.float32
        }
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=8), min_size=3, max_size=7))

        index_np = np.vstack(
            [np.random.randint(
                0, s, size=100) for s in in_shape]).T.astype(index_type)

        test_update_0d = draw(st.sampled_from([True, False]))
        if test_update_0d:
            in_shape = draw(
                st.lists(
                    st.integers(
                        min_value=2, max_value=8),
                    min_size=1,
                    max_size=1))
            index_np = np.array([1]).reshape([1]).astype(index_type)

        update_shape = judge_update_shape(in_shape, index_np.shape)

        if test_update_0d == False:
            assume(index_np.shape[-1] <= len(in_shape))

        def generate_data(*args, **kwargs):
            if kwargs["type"] == "int32":
                return np.random.randint(kwargs["low"], kwargs["high"],
                                         kwargs["shape"]).astype(np.int32)
            elif kwargs["type"] == "int64":
                return np.random.randint(kwargs["low"], kwargs["high"],
                                         kwargs["shape"]).astype(np.int64)
            elif kwargs["type"] == "float32":
                return np.random.random(kwargs["shape"]).astype(np.float32)

        def generate_index_data(*args, **kwargs):
            return index_np

        scatter_nd_add_op = OpConfig(
            type="scatter_nd_add",
            inputs={
                "X": ["input_data"],
                "Index": ["index"],
                "Updates": ["updates"]
            },
            outputs={"Out": ["output_data"]},
            outputs_dtype={"output_data": out_dtype_dict[input_type]},
            attrs={})

        program_config = ProgramConfig(
            ops=[scatter_nd_add_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(
                    generate_data,
                    type=input_type,
                    low=-10,
                    high=10,
                    shape=in_shape)),
                "index": TensorConfig(data_gen=partial(generate_index_data)),
                "updates": TensorConfig(data_gen=partial(
                    generate_data,
                    type=input_type,
                    low=-10,
                    high=10,
                    shape=update_shape)),
            },
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["scatter_nd_add"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 25
        if target_str == "Host":
            # Make sure to generate enough valid cases for Host
            max_examples = 400
        self.run_and_statis(
            quant=False, min_success_num=25, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
