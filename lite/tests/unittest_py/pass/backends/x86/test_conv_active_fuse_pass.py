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

import abc
import test_conv_active_fuse_pass_base
from auto_scan_test import FusePassAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st

class TestConvActiveFusePass(FusePassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self, *args, **kwargs):
        return test_conv_active_fuse_pass_base.sample_program_configs(*args, **kwargs)

    def sample_predictor_configs(self):
        config = CxxConfig()
        config.set_valid_places({Place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW)})
        yield config, (1e-5, 1e-5)

    def add_skip_pass_case(self):
        pass

    @given(
        in_shape=st.lists(st.integers(min_value=1, max_value=64), min_size=4, max_size=4),
        weight_shape=st.lists(st.integers(min_value=1, max_value=64), min_size=4, max_size=4),
        paddings=st.sampled_from([[1, 2], [4, 2]]),
        dilations=st.sampled_from([[1, 1]]),
        groups=st.sampled_from([1]),
        padding_algorithm=st.sampled_from(["VALID", "SAME"]),
        strides=st.sampled_from([[1, 1], [2, 2]]),
        threshold=st.floats(min_value=0, max_value=1),
        alpha=st.floats(min_value=0, max_value=1),
        scale=st.floats(min_value=0.5, max_value=5),
        offset=st.floats(min_value=0, max_value=1),
        )
    def test(self, *args, **kwargs):
        assume(kwargs["in_shape"][1] == kwargs["weight_shape"][1])
        assume(kwargs["in_shape"][2] >= kwargs["weight_shape"][2])
        assume(kwargs["in_shape"][3] >= kwargs["weight_shape"][3])
        self.add_skip_pass_case()
        optimized_model = self.run_test(quant=False, *args, **kwargs)
        self.assert_op_size(4, 3, self.origin_model, optimized_model)

if __name__ == "__main__":
    unittest.main()
