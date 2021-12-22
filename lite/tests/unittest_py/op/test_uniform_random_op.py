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
import random
import numpy as np


class TestUniformRandomOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)

        host_places = [
            Place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=host_places, thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        N = draw(st.integers(min_value=1, max_value=4))
        C = draw(st.integers(min_value=1, max_value=128))
        H = draw(st.integers(min_value=1, max_value=128))
        W = draw(st.integers(min_value=1, max_value=128))
        shape_data = draw(st.sampled_from([[N, C, H, W], [N, H, W]]))

        def generate_ShapeTensor():
            return np.array(shape_data).astype(np.int64)  # must be int64

        min_data = draw(st.floats(min_value=-1, max_value=-1))
        max_data = draw(st.floats(min_value=1, max_value=1))
        seed_data = draw(st.integers(min_value=1, max_value=1))
        dtype_data = draw(st.integers(
            min_value=5, max_value=5))  # out is float

        choose_shape = draw(
            st.sampled_from(["shape", "ShapeTensor", "ShapeTensorList"]))
        inputs = {}

        def generate_ShapeTensor_data():
            if (choose_shape == "ShapeTensor"):
                inputs["ShapeTensor"] = ["ShapeTensor_data"]
                return np.array(shape_data).astype(np.int64)
            else:
                return np.random.randint(1, 5, []).astype(np.int64)

        def generate_ShapeTensorList_data():
            if (choose_shape == "ShapeTensorList"):
                # TensorList is not supported by lite
                # inputs["ShapeTensorList"] : ["ShapeTensorList_data"]
                return np.array(shape_data).astype(np.int64)
            else:
                return np.random.randint(1, 5, []).astype(np.int64)

        uniform_random_op = OpConfig(
            type="uniform_random",
            inputs=inputs,
            outputs={"Out": ["output_data"]},
            attrs={
                "shape": shape_data,
                "min": min_data,
                "max": max_data,
                "seed": seed_data,
                "dtype": dtype_data,
                # lite does not use these 3 attr
                # so I default them
                "diag_num": 0,
                "diag_step": 0,
                "diag_val": 1.0,
            })
        program_config = ProgramConfig(
            ops=[uniform_random_op],
            weights={},
            inputs={
                "ShapeTensor_data":
                TensorConfig(data_gen=partial(generate_ShapeTensor_data)),
                "ShapeTensorList_data":
                TensorConfig(data_gen=partial(generate_ShapeTensorList_data))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), [""], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
