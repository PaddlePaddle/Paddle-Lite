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
import argparse
import numpy as np
from functools import partial


class TestLogicalOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.Host, PrecisionType.Any,
                                     DataLayoutType.NCHW)
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
                    min_value=3, max_value=64), min_size=2, max_size=4))
        op_type_str = draw(
            st.sampled_from(
                ["logical_and", "logical_not", "logical_or", "logical_xor"]))

        if self.get_nnadapter_device_name() == "kunlunxin_xtcl":
            in_shape = [1]
            assume(op_type_str == "logical_and" or
                   op_type_str == "logical_not")

        def generate_input_x():
            return np.random.choice(a=[0, 1], size=in_shape).astype(np.int32)

        def generate_input_y():
            return np.random.choice(a=[0, 1], size=in_shape).astype(np.int32)

        cast_x = OpConfig(
            type="cast",
            inputs={"X": ["input_data_x"], },
            outputs={"Out": ["cast_data_x"], },
            attrs={"in_dtype": int(2),
                   "out_dtype": int(0)})
        cast_x.outputs_dtype = {"cast_data_x": np.bool_}

        cast_y = OpConfig(
            type="cast",
            inputs={"X": ["input_data_y"], },
            outputs={"Out": ["cast_data_y"], },
            attrs={"in_dtype": int(2),
                   "out_dtype": int(0)})
        cast_y.outputs_dtype = {"cast_data_y": np.bool_}

        # two args 
        build_ops = OpConfig(
            type=op_type_str,
            inputs={"X": ["cast_data_x"],
                    "Y": ["cast_data_y"]},
            outputs={"Out": ["output_data"], },
            attrs={})
        build_ops.outputs_dtype = {"output_data": np.bool_}

        #one args
        build_op = OpConfig(
            type="logical_not",
            inputs={"X": ["cast_data_x"]},
            outputs={"Out": ["output_data"], },
            attrs={})
        build_op.outputs_dtype = {"output_data": np.bool_}

        cast_out = OpConfig(
            type="cast",
            inputs={"X": ["output_data"], },
            outputs={"Out": ["cast_data_out"], },
            attrs={"in_dtype": int(0),
                   "out_dtype": int(2)})
        cast_out.outputs_dtype = {"cast_data_out": np.int32}

        tmp_ops = []
        if op_type_str == "logical_not":
            tmp_ops = [cast_x, build_op, cast_out]
        else:
            tmp_ops = [cast_x, cast_y, build_ops, cast_out]
        program_config = ProgramConfig(
            ops=tmp_ops,
            weights={},
            inputs={
                "input_data_x":
                TensorConfig(data_gen=partial(generate_input_x)),
                "input_data_y":
                TensorConfig(data_gen=partial(generate_input_y)),
            },
            outputs=["cast_data_out"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["logical"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=60)


if __name__ == "__main__":
    unittest.main(argv=[''])
