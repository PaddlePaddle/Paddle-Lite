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
import hypothesis.strategies as st

from functools import partial
import numpy as np
import argparse


class TestReduceProdOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.X86, PrecisionType.FP32,
                                     DataLayoutType.NCHW)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
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
                    min_value=1, max_value=10), min_size=4, max_size=4))
        in_shape = draw(st.sampled_from([in_shape, []]))
        keep_dim = draw(st.booleans())
        axis_list = draw(
            st.sampled_from([[0], [1], [2], [3], [0, 1], [1, 2], [2, 3]]))

        reduce_all_data = True if len(
            in_shape) == 0 or axis_list == None or axis_list == [] else False

        if len(in_shape) == 0:
            axis_list = draw(st.sampled_from([[-1], []]))

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        build_ops = OpConfig(
            type="reduce_prod",
            inputs={"X": ["input_data"], },
            outputs={"Out": ["output_data"], },
            attrs={
                "dim": axis_list,
                "keep_dim": keep_dim,
                "reduce_all": reduce_all_data,
            })
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input)),
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["reduce_prod"], (1e-2, 1e-2)

    def add_ignore_pass_case(self):
        def _teller3(program_config, predictor_config):
            target_type = predictor_config.target()
            if target_type == TargetType.OpenCL:
                return True

        self.add_ignore_check_case(_teller3,
                                   IgnoreReasons.PADDLELITE_NOT_SUPPORT,
                                   "Expected kernel_type false.")

        def _teller4(program_config, predictor_config):
            target_type = predictor_config.target()
            if target_type not in [
                    TargetType.ARM, TargetType.Host, TargetType.X86,
                    TargetType.Metal, TargetType.OpenCL
            ]:
                if program_config.attr["reduce_all"]:
                    return True

        self.add_ignore_check_case(
            _teller4, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "0D-tensor is not supported on this target now.")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
