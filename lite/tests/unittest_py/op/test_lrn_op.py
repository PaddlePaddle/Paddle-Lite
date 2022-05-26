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


class TestLrnOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        # opencl bug will be fix in the future
        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.FP16,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.OpenCL, PrecisionType.Any,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.Any,
                      DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=opencl_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        #simple case
        '''
        in_shape = draw(st.sampled_from([[1, 5, 2, 4]]))
        n_ = draw(st.sampled_from([5]))
        k_ = draw(st.sampled_from([1.0]))
        alpha_ = draw(st.sampled_from([1e-4]))
        beta_ = draw(st.sampled_from([0.75, 1]))
        '''
        in_num = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=4), min_size=1, max_size=1))
        in_c_h_w = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=128),
                min_size=3,
                max_size=3))
        in_shape = in_num + in_c_h_w
        n_ = draw(st.integers(min_value=1, max_value=in_c_h_w[0]))
        k_ = draw(st.floats(min_value=1.0, max_value=10.0))
        alpha_ = draw(st.floats(min_value=1.0, max_value=10.0))
        beta_ = draw(st.floats(min_value=1.0, max_value=10.0))
        assume(n_ % 2 == 1)
        lrn_op = OpConfig(
            type="lrn",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"],
                     "MidOut": ["output_data_mid"]},
            attrs={
                "n": n_,
                "k": k_,
                "alpha": alpha_,
                "beta": beta_,
                "is_test": 1
            })
        program_config = ProgramConfig(
            ops=[lrn_op],
            weights={},
            inputs={"input_data": TensorConfig(shape=in_shape)},
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["lrn"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=150)


if __name__ == "__main__":
    unittest.main(argv=[''])
