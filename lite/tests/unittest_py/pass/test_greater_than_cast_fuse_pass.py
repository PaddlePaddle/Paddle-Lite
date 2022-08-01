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
sys.path.append('..')

from auto_scan_test import FusePassAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestGreaterThanCastFusePass(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        #opencl
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
        cast_in_type = draw(st.sampled_from([0]))
        cast_out_type = draw(st.sampled_from([2, 3, 5]))  #0,1,4 cannot

        in_shape_x = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=64), min_size=4, max_size=4))
        in_shape_y = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=64), min_size=1, max_size=1))

        def generate_input_x(*args, **kwargs):
            return np.random.randint(in_shape_x).astype(np.float32)

        def generate_input_y(*args, **kwargs):
            return np.random.randint(in_shape_y).astype(np.float32)

        greater_than_op = OpConfig(
            type="greater_than",
            inputs={"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["greater_than_output"]},
            attrs={"axis": -1,
                   "force_cpu": False})

        cast_op = OpConfig(
            type="cast",
            inputs={"X": ["greater_than_output"]},
            outputs={"Out": ["output_data"]},
            attrs={"in_dtype": cast_in_type,
                   "out_dtype": cast_out_type})

        ops = [greater_than_op, cast_op]
        program_config = ProgramConfig(
            ops=ops,
            weights={},
            inputs={
                "input_data_x":
                TensorConfig(data_gen=partial(generate_input_x)),
                "input_data_y":
                TensorConfig(data_gen=partial(generate_input_y))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ['greater_than'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False,
            max_examples=25,
            passes=["lite_greater_than_cast_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
