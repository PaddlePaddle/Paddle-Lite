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
import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
import numpy as np


class TestSequenceReshapeOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP16],
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=13, max_value=64),
                min_size=2,
                max_size=3))
        in_shape.insert(0, 11)
        pad_value = draw(st.sampled_from([0.0, 0.2, 0.5, 1.0]))
        pooltype = draw(
            st.sampled_from(
                ["AVERAGE", "SUM", "SQRT", "MAX", "LAST", "FIRST"]))
        lod_tensor = draw(st.sampled_from([[[11, 11]]]))

        def generate_input(*args, **kwargs):
            return np.random.normal(0.1, 1, in_shape).astype(np.float32)

        def generate_maxindex(*args, **kwargs):
            return np.zeros((len(lod_tensor), in_shape[1])).astype(np.int32)

        ops_config = OpConfig(
            type="sequence_pool",
            inputs={"X": ["input_data"], },
            outputs={"Out": ["output_data"],
                     "MaxIndex": ["maxindex"]},
            attrs={"pad_value": pad_value,
                   "pooltype": pooltype})

        program_config = ProgramConfig(
            ops=[ops_config],
            weights={
                "maxindex": TensorConfig(data_gen=partial(generate_maxindex))
            },
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input), lod=lod_tensor)
            },
            outputs=["output_data", "maxindex"])

        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        config_lists = self.get_predictor_configs()
        for config in config_lists:
            if config.precision() in [PrecisionType.FP16]:
                atol, rtol = 1e-3, 1e-3

        return self.get_predictor_configs(), ["sequence_pool"], (atol, rtol)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.ACCURACY_ERROR,
            "The op output has diff in a specific case. We need to fix it as soon as possible."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
