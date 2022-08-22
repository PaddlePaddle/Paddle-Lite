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


class TestWhereOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        host_places = [
            Place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=host_places)
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=["kunlunxin_xtcl"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=50), min_size=1, max_size=1))
        in_dtype = draw(st.sampled_from([np.float32, np.int32, np.int64]))

        if self.get_nnadapter_device_name() == "kunlunxin_xtcl":
            in_shape = [1]

        def generate_X_data():
            return np.random.normal(0.0, 5.0, in_shape).astype(in_dtype)

        def generate_Condition_data():
            return np.random.choice(
                [0, 1], in_shape, replace=True).astype(np.int32)

        cast_op = OpConfig(
            type="cast",
            inputs={"X": ["Condition_data"]},
            outputs={"Out": ["middle_data"]},
            attrs={
                "in_dtype": 2,  #int32
                "out_dtype": 0,  # bool
            })

        cast_op.outputs_dtype = {"middle_data": np.bool}

        where_op = OpConfig(
            type="where",
            inputs={
                "X": ["X_data"],
                "Y": ["Y_data"],
                "Condition": ["middle_data"]
            },
            outputs={"Out": ["Out_data"]},
            attrs={})

        where_op.outputs_dtype = {"Out_data": in_dtype}

        program_config = ProgramConfig(
            ops=[cast_op, where_op],
            weights={},
            inputs={
                "X_data": TensorConfig(data_gen=partial(generate_X_data)),
                "Y_data": TensorConfig(data_gen=partial(generate_X_data)),
                "Condition_data":
                TensorConfig(data_gen=partial(generate_Condition_data))
            },
            outputs=["Out_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), [""], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        max_examples = 25
        if self.get_nnadapter_device_name() == "kunlunxin_xtcl":
            max_examples = 200
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
