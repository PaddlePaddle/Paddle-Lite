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

import numpy as np
from functools import partial
import hypothesis.strategies as st


class TestLstmOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        # paddle crash bugs will be fixed in the future
        #self.enable_testing_on_place(
        #    TargetType.ARM,
        #    PrecisionType.FP32,
        #    DataLayoutType.NCHW,
        #    thread=[1, 4])
        #self.enable_testing_on_place(
        #    TargetType.X86,
        #    PrecisionType.FP32,
        #    DataLayoutType.NCHW,
        #    thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        shape0 = draw(st.integers(min_value=1, max_value=32))
        shape1 = draw(st.integers(min_value=1, max_value=32))
        shape2 = draw(st.integers(min_value=1, max_value=32))
        # lstm run have bugs, for debug
        #shape0 = 2
        #shape1 = 3
        #shape2 = 2
        input_lod_ = draw(st.sampled_from([[[shape0, shape1, shape2]]]))
        N = len(input_lod_[0])
        D = 16
        T = sum(input_lod_[0])
        in_shape = draw(st.sampled_from([[T, 4 * D]]))
        input_weight_shape = draw(st.sampled_from([[D, D * 4]]))
        input_h0_data_shape = draw(st.sampled_from([[N, D]]))
        input_c0_data_shape = draw(st.sampled_from([[N, D]]))
        input_bias_shape = draw(st.sampled_from([[1, D * 4]]))

        def generate_input_data_in_shape(*args, **kwargs):
            return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)

        def generate_input_data_input_weight_shape(*args, **kwargs):
            return np.random.normal(0.0, 1.0,
                                    input_weight_shape).astype(np.float32)

        def generate_input_data_input_h0_data_shape(*args, **kwargs):
            return np.random.normal(0.0, 1.0,
                                    input_h0_data_shape).astype(np.float32)

        def generate_input_data_input_c0_data_shape(*args, **kwargs):
            return np.random.normal(0.0, 1.0,
                                    input_c0_data_shape).astype(np.float32)

        def generate_input_data_input_bias_shape(*args, **kwargs):
            return np.random.normal(0.0, 1.0,
                                    input_bias_shape).astype(np.float32)

        use_p_ = draw(st.sampled_from([0]))
        is_r_ = draw(st.sampled_from([0]))
        gate_activation_ = draw(st.sampled_from(["sigmoid"]))
        cell_activation_ = draw(st.sampled_from(["tanh"]))
        candidate_activation_ = draw(st.sampled_from(["tanh"]))
        lstm_op = OpConfig(
            type="lstm",
            inputs={
                "Input": ["input_data"],
                "H0": ["input_h0_data"],
                "C0": ["input_c0_data"],
                "Weight": ["input_weight_data"],
                "Bias": ["input_bias_data"]
            },
            outputs={
                "Hidden": ["output_data_hidden"],
                "Cell": ["output_data_cell"],
                "BatchGate": ["BatchGate"],
                "BatchCellPreAct": ["BatchCellPreAct"]
            },
            attrs={
                "use_peepholes": use_p_,
                "is_reverse": is_r_,
                "gate_activation": gate_activation_,
                "cell_activation": cell_activation_,
                "candidate_activation": candidate_activation_,
                "is_test": 1
            })
        program_config = ProgramConfig(
            ops=[lstm_op],
            weights={},
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input_data_in_shape),
                    lod=input_lod_),
                "input_weight_data": TensorConfig(
                    data_gen=partial(generate_input_data_input_weight_shape)),
                "input_h0_data": TensorConfig(
                    data_gen=partial(generate_input_data_input_h0_data_shape)),
                "input_c0_data": TensorConfig(
                    data_gen=partial(generate_input_data_input_c0_data_shape)),
                "input_bias_data": TensorConfig(
                    data_gen=partial(generate_input_data_input_bias_shape))
            },
            outputs={
                "output_data_hidden", "output_data_cell", "BatchCellPreAct",
                "BatchGate"
            })
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["lstm"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
