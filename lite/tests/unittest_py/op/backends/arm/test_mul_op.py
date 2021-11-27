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

import test_mul_op_base
from auto_scan_test_rpc import AutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st

class TestMulOp(AutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self, *args, **kwargs):
        return test_mul_op_base.sample_program_configs(*args, **kwargs)

    def sample_predictor_configs(self):
        config = CxxConfig()
        config.set_valid_places({Place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW)})
        config.set_threads(1)
        yield config, (1e-5, 1e-5)

    def add_skip_pass_case(self):
        pass

    @given(
        in_shape=st.lists(
            st.integers(
                min_value=2, max_value=2), min_size=2, max_size=2))
    def test(self, *args, **kwargs):
        self.add_skip_pass_case()
        self.run_test(quant=False, *args, **kwargs)

if __name__ == "__main__":
    unittest.main()
