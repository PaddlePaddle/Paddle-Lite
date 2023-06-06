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


class TestNotEqualOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=["cambricon_mlu"])
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:

        return True

    def sample_program_configs(self, draw):
        in_shape_x = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=4, max_size=4))
        in_shape_x = draw(st.sampled_from([in_shape_x, []]))
        in_shape_y = in_shape_x
        axis = draw(st.sampled_from([0, 1, 2, 3]))
        if in_shape_x == []:
            axis = draw(st.sampled_from([0, -1]))
        not_equal_op = OpConfig(
            type="not_equal",
            inputs={"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["output_data"]},
            attrs={"axis": axis})
        cast_out = OpConfig(
            type="cast",
            inputs={"X": ["output_data"], },
            outputs={"Out": ["cast_data_out"], },
            attrs={"in_dtype": int(0),
                   "out_dtype": int(2)})
        cast_out.outputs_dtype = {"cast_data_out": np.int32}
        program_config = ProgramConfig(
            ops=[not_equal_op, cast_out],
            weights={},
            inputs={
                "input_data_x": TensorConfig(shape=in_shape_x),
                "input_data_y": TensorConfig(shape=in_shape_y)
            },
            outputs=["cast_data_out"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["not_equal"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
