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

from auto_scan_test import FusePassAutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
from test_elementwise_util import trim_trailing_singular_dims, check_input_shape_available
import hypothesis.strategies as st


class TestElementwiseScaleFuse(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        N = draw(st.integers(min_value=1, max_value=4))
        C = draw(st.integers(min_value=1, max_value=128))
        H = draw(st.integers(min_value=1, max_value=128))
        W = draw(st.integers(min_value=1, max_value=128))
        in_shape = draw(st.sampled_from([[N, C, H, W], []]))
        in_dtype = draw(st.sampled_from([np.float32, np.int32, np.int64]))

        def generate_X_data():
            return np.random.normal(0.0, 5.0, in_shape).astype(in_dtype)

        axes_data = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=3), min_size=1, max_size=2))
        if in_shape == []:
            axes_data = [0]

        inputs = {"X": ["X_data"]}
        choose_axes = draw(
            st.sampled_from(["axes", "AxesTensor", "AxesTensorList"]))

        def generate_AxesTensor_data():
            if (choose_axes == "AxesTensor"):
                inputs["AxesTensor"] = ["AxesTensor_data"]
                if in_shape == []:
                    return np.array([0]).astype(np.int32)
                else:
                    return np.array(axes_data).astype(np.int32)
            else:
                return np.random.randint(1, 5, []).astype(np.int32)

        def generate_AxesTensorList_data():
            if (choose_axes == "AxesTensorList"):
                #inputs["AxesTensorList"] = ["AxesTensorList_data"]
                if in_shape == []:
                    return np.array([0]).astype(np.int32)
                else:
                    return np.array(axes_data).astype(np.int32)
            else:
                return np.random.randint(1, 5, []).astype(np.int32)

        def generate_XShape_data():
            return np.random.random([6]).astype(np.float32)

        unsqueeze2_op = OpConfig(
            type="unsqueeze2",
            inputs=inputs,
            outputs={"Out": ["Out_data"],
                     "XShape": ["XShape_data"]},
            attrs={"axes": axes_data, })
        unsqueeze2_op.outputs_dtype = {"Out_data": in_dtype}

        program_config = ProgramConfig(
            ops=[unsqueeze2_op],
            weights={
                "XShape_data":
                TensorConfig(data_gen=partial(generate_XShape_data))
            },
            inputs={
                "X_data": TensorConfig(data_gen=partial(generate_X_data)),
                "AxesTensor_data":
                TensorConfig(data_gen=partial(generate_AxesTensor_data)),
                "AxesTensorList_data":
                TensorConfig(data_gen=partial(generate_AxesTensorList_data))
            },
            outputs=["Out_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ['unsqueeze2'], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 100
        if target_str == "OpenCL":
            max_examples = 1000
        self.run_and_statis(
            quant=False,
            max_examples=max_examples,
            passes=["unsqueeze_calc_offline_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
