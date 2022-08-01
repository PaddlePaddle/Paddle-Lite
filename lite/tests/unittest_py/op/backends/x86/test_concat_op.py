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
sys.path.append('../../common')
sys.path.append('../../../')

import test_concat_op_base
from auto_scan_test import AutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume


class TestConcatOp(AutoScanTest):
    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        in_shape1 = program_config.inputs["input_data1"].shape
        in_shape2 = program_config.inputs["input_data2"].shape
        len1 = len(in_shape1)
        len2 = len(in_shape2)
        axis = program_config.ops[0].attrs["axis"]
        if axis < 0:
            axis += len1

        if len1 != len2:
            return False
        for i in range(0, len1):
            if i == axis:
                continue
            if in_shape1[i] != in_shape2[i]:
                return False

        if axis >= len1:
            return False
        else:
            return True

    def sample_program_configs(self, draw):
        return test_concat_op_base.sample_program_configs(draw)

    def sample_predictor_configs(self):
        config = CxxConfig()
        config.set_valid_places(
            {Place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW)})
        yield config, ["concat"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
