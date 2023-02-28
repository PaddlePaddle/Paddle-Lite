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
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st
import numpy as np
from functools import partial


class TestEqualOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            [PrecisionType.FP32, PrecisionType.INT64, PrecisionType.INT32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        metal_places = [
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.Metal, PrecisionType.FP16,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.ARM, PrecisionType.FP32),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=metal_places)
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=[
            "cambricon_mlu", "intel_openvino", "kunlunxin_xtcl"
        ])
        # "nvidia_tensorrt" removed by zhoukangkang

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        target_type = predictor_config.target()
        in_x_shape = list(program_config.inputs["input_data_x"].shape)
        input_data_type = program_config.inputs["input_data_x"].dtype
        # check config
        if predictor_config.precision(
        ) == PrecisionType.INT64 and input_data_type != np.int64:
            return False
        if predictor_config.precision(
        ) == PrecisionType.FP32 and input_data_type != np.float32:
            return False
        if predictor_config.precision(
        ) == PrecisionType.INT32 and input_data_type != np.int32:
            return False

        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=1, max_size=6))
        if self.get_target() == "Metal":
            in_shape = draw(
                st.lists(
                    st.integers(
                        min_value=1, max_value=8),
                    min_size=3,
                    max_size=3))
            in_shape.insert(0, 1)

        data_type = draw(st.sampled_from(['float32', 'int32', 'int64']))

        def gen_input_data(*args, **kwargs):
            if kwargs['dtype'] == 'float32':
                return np.random.normal(0.0, 1.0,
                                        kwargs['shape']).astype(np.float32)
            elif kwargs['dtype'] == 'int32':
                return np.random.randint(1, 40,
                                         kwargs['shape']).astype(np.int32)
            elif kwargs['dtype'] == 'int64':
                return np.random.randint(1, 40,
                                         kwargs['shape']).astype(np.int64)

        equal_op = OpConfig(
            type="equal",
            inputs={"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["output_data"]},
            attrs={})
        cast_op = OpConfig(
            type="cast",
            inputs={"X": ["output_data"]},
            outputs={"Out": ["cast_output_data"]},
            attrs={'in_dtype': 0,
                   'out_dtype': 5})  # 0: bool , 5: float
        program_config = ProgramConfig(
            ops=[equal_op, cast_op],
            weights={},
            inputs={
                "input_data_x": TensorConfig(data_gen=partial(
                    gen_input_data, dtype=data_type, shape=in_shape)),
                "input_data_y": TensorConfig(data_gen=partial(
                    gen_input_data, dtype=data_type, shape=in_shape))
            },
            outputs=["cast_output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["equal"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            # All outputs are zero. But diff occurred.
            if predictor_config.target() == TargetType.Metal:
                return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.ACCURACY_ERROR,
            "The op output has diff in a specific case. We need to fix it as soon as possible."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
