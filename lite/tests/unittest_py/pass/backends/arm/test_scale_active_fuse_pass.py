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

import test_scale_active_fuse_pass_base
from auto_scan_test_rpc import FusePassAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st

class TestScaleActiveFusePass(FusePassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig , predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        return test_scale_active_fuse_pass_base.sample_program_configs(draw)

    def sample_predictor_configs(self):
        config = CxxConfig()
        config.set_valid_places({Place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW)})
        yield config, ["scale"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25, passes=["lite_scale_activation_fuse_pass"])

if __name__ == "__main__":
    unittest.main(argv=[''])
