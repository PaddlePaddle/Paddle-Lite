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


class TestGruOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1])
        #self.enable_testing_on_place(TargetType.ARM, 
        # PrecisionType.INT8, DataLayoutType.NCHW, thread=[1, 4])

        # precision diff
        # self.enable_testing_on_place(
        #     TargetType.ARM,
        #     PrecisionType.FP16,
        #     DataLayoutType.NCHW,
        #     thread=[1, 2, 4])
        
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 2, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        x_dtype = program_config.inputs["input_data"].dtype
        if x_dtype != np.float32:
            if predictor_config.target() == TargetType.ARM:
                return True
            else:
                return False
        return True

    def sample_program_configs(self, draw):
        is_rev = draw(st.sampled_from([False, True]))
        bool_orimode = draw(st.sampled_from([True, False]))
        in_shape = draw(
            st.sampled_from([[20, 60], [30, 90], [40, 120], [50, 150],
                             [60, 180]]))
        h0_1 = draw(st.sampled_from([3]))
        process_type = draw(st.sampled_from(
            ["type_fp32"]))  #int8 to be supported

        #FP32
        def generate_input(*args, **kwargs):
            return np.random.random([9, in_shape[1]]).astype(np.float32)

        def generate_weight(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_bias(*args, **kwargs):
            return np.random.random([1, in_shape[1]]).astype(np.float32)

        def generate_h0(*args, **kwargs):
            return np.random.random([h0_1, in_shape[0]]).astype(np.float32)

        #INT8, to be supported
        def generate_input_int8(*args, **kwargs):
            return np.random.randint(
                low=0, high=10, size=[9, in_shape[1]]).astype(np.int8)

        def generate_weight_int8(*args, **kwargs):
            return np.random.randint(
                low=0, high=10, size=in_shape).astype(np.int8)

        def generate_bias_int8(*args, **kwargs):
            return np.random.randint(
                low=0, high=10, size=[1, in_shape[1]]).astype(np.int8)

        def generate_h0_int8(*args, **kwargs):
            return np.random.randint(
                low=0, high=10, size=[h0_1, in_shape[0]]).astype(np.int8)

        def generate_scale_w(*args, **kwargs):
            return np.random.random([in_shape[0]]).astype(np.float32)

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

        if process_type == "type_fp32":
            program_config = ProgramConfig(
                ops=[build_ops],
                weights={
                    "weight_data":
                    TensorConfig(data_gen=partial(generate_weight)),
                },
                inputs={
                    "input_data": TensorConfig(
                        data_gen=partial(generate_input), lod=[[0, 2, 6, 9]]),
                    "bias_data": TensorConfig(data_gen=partial(generate_bias)),
                    "h0": TensorConfig(data_gen=partial(generate_h0)),
                },
                outputs=["hidden"])
        elif process_type == "type_int8":
            build_op = OpConfig(
                type="gru",
                inputs={
                    "Input": ["input_data"],
                    "Weight": ["weight_data"],
                    "Bias": ["bias_data"],
                    "H0": ["h0"],
                    "Weight0_scale": ["scale_w"],
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
                    "is_test": True,
                    "enable_int8": True,
                    "bit_length": 8
                })
            build_op.outputs_dtype = {
                "hidden": np.int8,
                "batch_gate": np.int8,
                "batch_reset_hidden_prev": np.int8,
                "batch_hidden": np.int8
            }
            program_config = ProgramConfig(
                ops=[build_op],
                weights={
                    "weight_data":
                    TensorConfig(data_gen=partial(generate_weight_int8)),
                },
                inputs={
                    "input_data": TensorConfig(
                        data_gen=partial(generate_input_int8),
                        lod=[[0, 2, 6, 9]]),
                    "bias_data":
                    TensorConfig(data_gen=partial(generate_bias_int8)),
                    "h0": TensorConfig(data_gen=partial(generate_h0_int8)),
                    "scale_w":
                    TensorConfig(data_gen=partial(generate_scale_w)),
                },
                outputs=["hidden"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["gru"], (6e-4, 6e-4)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=60)


if __name__ == "__main__":
    unittest.main(argv=[''])
