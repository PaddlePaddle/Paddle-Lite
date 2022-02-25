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


class TestFcOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.X86, PrecisionType.FP32,
                                     DataLayoutType.NCHW)

        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.OpenCL, PrecisionType.Any,
                  DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=opencl_places)
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(
            device_names=["kunlunxin_xtcl", "cambricon_mlu"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=10), min_size=2, max_size=4))
        in_num_col_dims = draw(
            st.integers(
                min_value=1, max_value=len(in_shape) - 1))
        weights_0 = 1
        weights_1 = 1
        for i in range(len(in_shape)):
            if (i < in_num_col_dims):
                weights_1 = weights_1 * in_shape[i]
            else:
                weights_0 = weights_0 * in_shape[i]
        weights_shape = [weights_0, weights_1]
        padding_weights = draw(st.booleans())
        # OpenCL dose not support this attribute
        if (self.get_target() == 'OpenCL'):
            padding_weights = False
        if (padding_weights):
            weights_shape = [weights_0 + 4, weights_1 + 4]

        bias_shape = [weights_1]
        with_bias = draw(st.sampled_from([True, False]))

        def generate_input(*args, **kwargs):
            return (np.random.random(in_shape).astype(np.float32) - 0.5) * 2

        def generate_weights(*args, **kwargs):
            return (
                np.random.random(weights_shape).astype(np.float32) - 0.5) * 2

        def generate_bias(*args, **kwargs):
            return (np.random.random(bias_shape).astype(np.float32) - 0.5) * 2

        act_type = ""

        if (with_bias and random.random() > 0.5):
            act_type = "relu"

        op_inputs = {}
        program_inputs = {}

        if (with_bias):
            op_inputs = {
                "Input": ["input_data"],
                "W": ["weights_data"],
                "Bias": ["bias_data"]
            }
            program_inputs = {
                "input_data": TensorConfig(data_gen=partial(generate_input)),
                "weights_data":
                TensorConfig(data_gen=partial(generate_weights)),
                "bias_data": TensorConfig(data_gen=partial(generate_bias))
            }
        else:
            op_inputs = {"Input": ["input_data"], "W": ["weights_data"]}
            program_inputs = {
                "input_data": TensorConfig(data_gen=partial(generate_input)),
                "weights_data":
                TensorConfig(data_gen=partial(generate_weights))
            }

        fc_op = OpConfig(
            type="fc",
            inputs=op_inputs,
            outputs={"Out": ["output_data"]},
            attrs={
                "in_num_col_dims": in_num_col_dims,
                "activation_type": act_type,
                "use_mkldnn": False,
                "padding_weights": padding_weights,
                "use_quantizer": False,
                "Scale_in": float(1),
                "Scale_weights": [float(1)],
                "Scale_out": float(1)
            })
        program_config = ProgramConfig(
            ops=[fc_op],
            weights={},
            inputs=program_inputs,
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["fc"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
