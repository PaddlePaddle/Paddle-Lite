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
import argparse
import numpy as np
from functools import partial
import copy


class TestLookupTableDequantOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.Any,
            DataLayoutType.NCHW,
            thread=[1, 2, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=3, max_value=64), min_size=2, max_size=2))
        id_shape = draw(
            st.lists(
                st.integers(
                    min_value=3, max_value=32), min_size=2, max_size=4))
        pidx = draw(st.sampled_from([-1, 0, 1, 2]))
        op_type_str = draw(st.sampled_from(["lookup_table_dequant"]))

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_ids(*args, **kwargs):
            extend_id = copy.deepcopy(id_shape)
            extend_id.append(1)
            return np.random.randint(
                low=0, high=in_shape[0], size=extend_id).astype(np.int64)

        build_ops = OpConfig(
            type=op_type_str,
            inputs={
                "W": ["w_data"],
                "Ids": ["ids_data"],
            },
            outputs={"Out": ["output_data"], },
            attrs={"padding_idx": int(pidx), })
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs={
                "w_data": TensorConfig(data_gen=partial(generate_input)),
                "ids_data": TensorConfig(data_gen=partial(generate_ids)),
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["lookup_table_dequant"], (1e-5,
                                                                        1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
