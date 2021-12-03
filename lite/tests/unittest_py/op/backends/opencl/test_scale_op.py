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

import test_scale_op_base
from auto_scan_test import AutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume

class TestScaleOp(AutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        return test_scale_op_base.sample_program_configs(draw)

    def sample_predictor_configs(self):
        config = CxxConfig()
        config.set_valid_places({Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageDefault),
                                 Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageFolder),
                                 Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
                                 Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageDefault),
                                 Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageFolder),
                                 Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
                                 Place(TargetType.X86, PrecisionType.FP32),
                                 Place(TargetType.ARM, PrecisionType.FP32),
                                 Place(TargetType.Host, PrecisionType.FP32)})
        yield config, ["scale"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        # ScaleTensor is not supported on opencl
        def teller(program_config, predictor_config):
            if 'ScaleTensor' in program_config.inputs:
                print("ScaleTensor happened")
                return True
            return False

        self.add_ignore_check_case(
            teller, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Paddle Lite does not support ScaleTensor in scale op."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)

if __name__ == "__main__":
    unittest.main()
