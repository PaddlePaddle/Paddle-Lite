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


class TestReduceSumOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.X86, PrecisionType.FP32,
                                     DataLayoutType.NCHW)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(
            device_names=["nvidia_tensorrt", "kunlunxin_xtcl"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=4, max_size=4))
        keep_dim = draw(st.booleans())
        axis_list = draw(
            st.sampled_from([[-1], [-2], [-3], [-4], [-2, -1], [-3, -2], [0],
                             [1], [2], [3], [0, 1], [1, 2], [2, 3]]))

        reduce_all_data = True if axis_list == None or axis_list == [] else False

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        build_ops = OpConfig(
            type="reduce_sum",
            inputs={"X": ["input_data"], },
            outputs={"Out": ["output_data"], },
            attrs={
                "dim": axis_list,
                "keep_dim": keep_dim,
                "reduce_all": reduce_all_data,
                "out_dtype": 5,
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
        return self.get_predictor_configs(), ["reduce_sum"], (1e-2, 1e-2)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "nvidia_tensorrt":
                in_shape = program_config.inputs["input_data"].shape
                axis_list = program_config.ops[0].attrs["dim"]
                if len(in_shape) == 1 \
                    or 0 in axis_list \
                    or -len(in_shape) in axis_list:
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'in_shape_size == 1' or 'axis has 0' on nvidia_tensorrt."
        )

        def _teller3(program_config, predictor_config):
            target_type = predictor_config.target()
            if target_type == TargetType.OpenCL:
                return True

        self.add_ignore_check_case(_teller3,
                                   IgnoreReasons.PADDLELITE_NOT_SUPPORT,
                                   "Expected kernel_type false.")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=100)


if __name__ == "__main__":
    unittest.main(argv=[''])
