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
import argparse
import numpy as np
from functools import partial


class TestGruOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

        #not support
        # self.enable_testing_on_place(
        #     TargetType.ARM,
        #     PrecisionType.INT8,
        #     DataLayoutType.NCHW,
        #     thread=[1, 4])

        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP16,
            DataLayoutType.NCHW,
            thread=[1, 4])

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
        shape0 = draw(st.integers(min_value=1, max_value=3))
        shape1 = draw(st.integers(min_value=1, max_value=3))
        shape2 = draw(st.integers(min_value=1, max_value=3))
        lod_arr = [0, shape0, shape0 + shape1, shape0 + shape1 + shape2]

        is_rev = draw(st.sampled_from([False, True]))
        bool_orimode = draw(st.sampled_from([True, False]))
        shape_0 = draw(st.integers(min_value=1, max_value=60))
        in_shape = [shape_0, shape_0 * 3]
        batch_num = lod_arr[3]
        h0_1 = len(lod_arr) - 1

        def generate_input(*args, **kwargs):
            return np.random.random(
                [batch_num, in_shape[1]]).astype(np.float32)

        def generate_weight(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_bias(*args, **kwargs):
            return np.random.random([1, in_shape[1]]).astype(np.float32)

        def generate_h0(*args, **kwargs):
            return np.random.random([h0_1, in_shape[0]]).astype(np.float32)

        build_ops = OpConfig(
            type="gru",
            inputs={
                "Input": ["input_data"],
                "Weight": ["weight_data"],
                "Bias": ["bias_data"],
                "H0": ["h0"]
            },
            outputs={
                "Hidden": ["hidden"],
                "BatchGate": ["batch_gate"],
                "BatchResetHiddenPrev": ["batch_reset_hidden_prev"],
                "BatchHidden": ["batch_hidden"]
            },
            attrs={
                "activation": "tanh",
                "gate_activation": "sigmoid",
                "is_reverse": is_rev,
                "origin_mode": bool_orimode,
                "is_test": True
            })

        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input), lod=[lod_arr]),
                "bias_data": TensorConfig(data_gen=partial(generate_bias)),
                "h0": TensorConfig(data_gen=partial(generate_h0)),
                "weight_data": TensorConfig(data_gen=partial(generate_weight)),
            },
            outputs=["hidden"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["gru"], (6e-4, 6e-4)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            if predictor_config.target() == TargetType.ARM:
                if predictor_config.precision() == PrecisionType.INT8:
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support this op for precision int8 on ARM for now.")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
