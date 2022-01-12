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
            TargetType.X86, [PrecisionType.INT64],
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        # check config
        x_dtype = program_config.inputs["input_data"].dtype
        if predictor_config.precision() == PrecisionType.INT64:
            if x_dtype != np.int64:
                return False
        return True

    def sample_program_configs(self, draw):
        lod_data = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=32), min_size=0, max_size=3))
        lod_data.append(12)
        new_dim = draw(st.sampled_from([12]))
        input_type = draw(st.sampled_from(["float32", "int64"]))

        def generate_input(*args, **kwargs):
            if input_type == "float32":
                return np.random.normal(0.0, 6.0, [12, 12]).astype(np.float32)
            elif input_type == "int64":
                return np.random.normal(0.0, 6.0, [12, 12]).astype(np.int64)

        ops_config = OpConfig(
            type="sequence_reshape",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"]},
            attrs={"new_dim": new_dim})

        program_config = ProgramConfig(
            ops=[ops_config],
            weights={},
            inputs={
                "input_data": TensorConfig(
                    data_gen=partial(generate_input), lod=[lod_data])
            },
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["sequence_reshape"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=50)


if __name__ == "__main__":
    unittest.main(argv=[''])
