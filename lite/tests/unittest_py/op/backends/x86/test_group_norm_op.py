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

from test_group_norm_op_base import TestGroupNormOpBase
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st

class TestGroupNormOp(TestGroupNormOpBase):
    def sample_predictor_configs(self):
        config = CxxConfig()
        config.set_valid_places({Place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW)})
        yield config, (1e-5, 1e-5)
    
    def add_skip_pass_case(self):
        pass

    @given(
        in_shape=st.lists(
            st.integers(
                min_value=1, max_value=10), min_size=4, max_size=4),
        group = st.sampled_from([1, 2, 4])
    )
    def test(self, *args, **kwargs):
        assume(kwargs["in_shape"][1] % kwargs["group"] == 0)
        self.add_skip_pass_case()
        self.run_test(quant=False, *args, **kwargs)

if __name__ == "__main__":
    unittest.main()
