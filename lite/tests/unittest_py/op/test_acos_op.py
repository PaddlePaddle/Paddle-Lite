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
import numpy as np
from functools import partial
import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
import argparse


class TestAcosOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
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
        def generate_data(*args, **kwargs):
            low, high = -10, 10
            dtype = "float32"
            shape = kwargs["shape"]
            if "low" in kwargs:
                low = kwargs["low"]
            if "high" in kwargs:
                high = kwargs["high"]
            if "dtype" in kwargs:
                dtype = kwargs["dtype"]
            if shape == []:
                if dtype == "int32":
                    return np.ones(shape).astype(np.int32)
                elif dtype == "int64":
                    return np.ones(shape).astype(np.int64)
                elif dtype == "float32":
                    return np.random.random(shape).astype(np.float32)
            else:
                if dtype == "int32":
                    if low == high:
                        return low * np.ones(shape).astype(np.int32)
                    else:
                        return np.random.randint(low, high,
                                                 shape).astype(np.int32)
                elif dtype == "int64":
                    if low == high:
                        return low * np.ones(shape).astype(np.int64)
                    else:
                        return np.random.randint(low, high,
                                                 shape).astype(np.int64)
                elif dtype == "float32":
                    return (
                        high - low
                    ) * np.random.random(shape).astype(np.float32) + low

        in_shape_tmp = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=4, max_size=4))
        in_shape = draw(st.sampled_from([in_shape_tmp, []]))

        acos_op = OpConfig(
            type="acos",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"]},
            attrs={})
        program_config = ProgramConfig(
            ops=[acos_op],
            weights={},
            inputs={
                "input_data": TensorConfig(data_gen=partial(
                    generate_data, low=-1, high=1, shape=in_shape))
            },
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["acos"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            in_x_shape = list(program_config.inputs["input_data"].shape)
            if target_type not in [
                    TargetType.ARM, TargetType.Host, TargetType.X86,
                    TargetType.Metal, TargetType.OpenCL
            ]:
                if len(in_x_shape) == 0:
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "0D-tensor is not supported on this target now.")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
