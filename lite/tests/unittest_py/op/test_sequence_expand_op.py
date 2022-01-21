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


class TestSequenceExpandOp(AutoScanTest):
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
        def generate_input(*args, **kwargs):
            if kwargs["type"] == "int32":
                return np.random.randint(kwargs["low"], kwargs["high"],
                                         kwargs["shape"]).astype(np.int32)
            elif kwargs["type"] == "int64":
                return np.random.randint(kwargs["low"], kwargs["high"],
                                         kwargs["shape"]).astype(np.int64)
            elif kwargs["type"] == "float32":
                return (kwargs["high"] - kwargs["low"]) * np.random.random(
                    kwargs["shape"]).astype(np.float32) + kwargs["low"]

        def generate_lod(seq_num, max_len):
            seq_offset = []
            sum = 0
            seq_offset.append(sum)
            for i in range(seq_num):
                sum += np.random.randint(0, max_len) + 1
                seq_offset.append(sum)
            return [seq_offset]

        input_type = draw(st.sampled_from(["int32", "int64", "float32"]))
        max_len = draw(st.integers(min_value=1, max_value=5))
        ref_level_data = draw(st.integers(min_value=-1, max_value=4))
        seq_num = draw(st.integers(min_value=1, max_value=7))
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=8))
        lod_x = generate_lod(seq_num, max_len)
        lod_y = generate_lod(seq_num, max_len)
        in_shape[0] = lod_x[0][-1]

        x_lod_len = lod_x[0][-1]
        y_lod_len = lod_y[0][-1]
        assume(lod_x[0][0] == lod_y[0][0])
        # x lod level is <= 1; y lod level is > 0
        assume((np.array(lod_x)).shape[0] <= 1)
        assume((np.array(lod_y)).shape[0] > 0)
        assume(ref_level_data == -1 or
               (ref_level_data >= 0 and
                ref_level_data < (np.array(lod_y)).shape[0]))
        assume((np.array(lod_x)).shape[0] == 1 and
               x_lod_len == (len(lod_y[ref_level_data])))

        sequence_expand_op = OpConfig(
            type="sequence_expand",
            inputs={"X": ["x_data"],
                    "Y": ["y_data"]},
            outputs={"Out": ["output_data"]},
            attrs={"ref_level": ref_level_data})

        program_config = ProgramConfig(
            ops=[sequence_expand_op],
            weights={},
            inputs={
                "x_data": TensorConfig(
                    data_gen=partial(
                        generate_input,
                        type=input_type,
                        low=-10,
                        high=10,
                        shape=in_shape),
                    lod=lod_x),
                "y_data": TensorConfig(
                    data_gen=partial(
                        generate_input,
                        type=input_type,
                        low=-10,
                        high=10,
                        shape=in_shape),
                    lod=lod_y)
            },
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["sequence_expand"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
