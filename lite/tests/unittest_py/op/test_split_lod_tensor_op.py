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


class TestSplitOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP32],
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
                    min_value=0, max_value=1), min_size=1, max_size=20))
        level = draw(st.sampled_from([0]))
        count = 0
        for i in in_shape:
            count += i
        assume(count != 0)
        assume(count != len(in_shape))

        def generate_input1(*args, **kwargs):
            return np.arange(len(in_shape)).reshape(len(in_shape),
                                                    1).astype('int32')

        def generate_mask_np(*args, **kwargs):
            mask_np = np.array(in_shape).astype('int32')
            mask_np = np.expand_dims(mask_np, axis=1)
            return mask_np

        cast_mask_np = OpConfig(
            type="cast",
            inputs={"X": ["Mask_input"]},
            outputs={"Out": ["Mask_output"]},
            attrs={"in_dtype": int(2),
                   "out_dtype": int(0)})
        cast_mask_np.outputs_dtype = {"Mask_output": np.bool_}

        ops_config = OpConfig(
            type="split_lod_tensor",
            inputs={"X": ["input_data"],
                    "Mask": ["Mask_output"]},
            outputs={
                "OutTrue": ["output_true"],
                "OutFalse": ["output_false"]
            },
            attrs={"level": level})

        program_config = ProgramConfig(
            ops=[cast_mask_np, ops_config],
            weights={
                "Mask_input": TensorConfig(data_gen=partial(generate_mask_np))
            },
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input1))
            },
            outputs=["output_true", "output_false"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["split_lod_tensor"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
