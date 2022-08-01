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


class TestSequenceExpandAsOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        x86_places = [
            Place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=x86_places, thread=[1, 4])
        arm_places = [
            Place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=arm_places, thread=[1, 4])

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

        input_type = draw(st.sampled_from(["int32", "int64", "float32"]))
        x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=2))
        y_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=2))
        y_lod = generate_input(
            type="int64", low=1, high=10, shape=[x_shape[0] + 1])
        y_lod = list(set(y_lod))
        y_lod = np.sort(y_lod)
        y_lod[0] = 0
        assume(x_shape[0] == len(y_lod) - 1)

        sequence_expand_as_op = OpConfig(
            type="sequence_expand_as",
            inputs={"X": ["x_data"],
                    "Y": ["y_data"]},
            outputs={"Out": ["output_data"]},
            attrs={})

        program_config = ProgramConfig(
            ops=[sequence_expand_as_op],
            weights={},
            inputs={
                "x_data": TensorConfig(data_gen=partial(
                    generate_input,
                    type=input_type,
                    low=-10,
                    high=10,
                    shape=x_shape)),
                "y_data": TensorConfig(
                    data_gen=partial(
                        generate_input,
                        type=input_type,
                        low=-10,
                        high=10,
                        shape=y_shape),
                    lod=[y_lod])
            },
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["sequence_expand_as"], (1e-5,
                                                                      1e-5)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            if target_type == TargetType.ARM:
                if program_config.inputs["x_data"].dtype != "float32":
                    print(
                        "The input data type only support float32 on ARM impl. Skip!"
                    )
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.ACCURACY_ERROR,
            "The op output has diff in a specific case on arm. We need to fix it as soon as possible."
        )

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 25
        if target_str == "ARM":
            # Make sure to generate enough valid cases for ARM
            max_examples = 100
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
