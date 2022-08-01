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


class TestSequenceConvOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP16,
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        context_start = draw(st.sampled_from([-2, -1, 0]))
        context_stride = 1
        context_length = 3
        kernel_num = draw(st.integers(min_value=2, max_value=8))
        in_dims = draw(
            st.lists(
                st.integers(
                    min_value=4, max_value=100),
                min_size=2,
                max_size=2))
        filter_dims = [context_length * in_dims[1], kernel_num]
        lod_info = draw(st.sampled_from([[[0, 4]], [[0, 2, 4]]]))

        padding_trainable = draw(st.booleans())

        assume(context_stride == 1)
        assume(len(in_dims) == 2 and len(filter_dims) == 2)
        if padding_trainable:
            print('paddingTrainable == True is not supported by now.')
        assume(padding_trainable == False)

        def generate_input(*args, **kwargs):
            return np.random.random(in_dims).astype(np.float32)

        def generate_filter(*args, **kwargs):
            return np.random.random(filter_dims).astype(np.float32)

        def generate_padding(*args, **kwargs):
            begin_pad = np.max([0, -context_start])
            end_pad = np.max([0, context_start + context_length - 1])
            total_pad = begin_pad + end_pad
            return np.random.uniform(0.1, 1,
                                     [total_pad, in_dims[1]]).astype('float32')

        inputs_dict = {"X": ["input_data"], "Filter": ["filter_data"]}
        inputs_gen_dict = {
            "input_data": TensorConfig(
                data_gen=partial(generate_input), lod=lod_info)
        }
        if padding_trainable:
            inputs_dict["PaddingData"] = ["padding_data"]
            inputs_gen_dict["padding_data"] = TensorConfig(
                data_gen=partial(generate_padding))

        sequence_conv_op = OpConfig(
            type="sequence_conv",
            inputs=inputs_dict,
            outputs={"Out": ["output_data"]},
            attrs={
                "contextStart": context_start,
                "contextStride": context_stride,
                "contextLength": context_length,
                "paddingTrainable": padding_trainable
            })
        program_config = ProgramConfig(
            ops=[sequence_conv_op],
            weights={
                "filter_data": TensorConfig(data_gen=partial(generate_filter))
            },
            inputs=inputs_gen_dict,
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["sequence_conv"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
