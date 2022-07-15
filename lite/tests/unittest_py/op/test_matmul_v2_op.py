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

from os import terminal_size
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


class TestMatmulV2Op(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.ARM, PrecisionType.FP32,
                                     DataLayoutType.NCHW)
        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageFolder), Place(
                      TargetType.OpenCL, PrecisionType.FP16,
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
        metal_places = [
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.Metal, PrecisionType.FP16,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.ARM, PrecisionType.FP32),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(places=opencl_places)
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(
            device_names=["nvidia_tensorrt", "intel_openvino"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        target_str = self.get_target()
        persistable = draw(st.booleans())
        if target_str == "OpenCL":
            shape0 = draw(st.integers(min_value=1, max_value=4)) * 4
            shape1 = draw(st.integers(min_value=1, max_value=4)) * 4
            shape2 = draw(st.integers(min_value=1, max_value=4)) * 4
            channels = draw(st.integers(min_value=1, max_value=64))
            batch = draw(st.integers(min_value=1, max_value=4))
        if target_str == "ARM" or target_str == "X86" or target_str == "NNAdapter":
            shape0 = draw(st.integers(min_value=1, max_value=64))
            shape1 = draw(st.integers(min_value=1, max_value=64))
            shape2 = draw(st.integers(min_value=1, max_value=64))
            channels = draw(st.integers(min_value=1, max_value=64))
            batch = draw(st.integers(min_value=1, max_value=4))
            assume(persistable == True)
        if target_str == "Metal":
            shape0 = draw(st.integers(min_value=1, max_value=64))
            shape1 = draw(st.integers(min_value=1, max_value=64))
            shape2 = draw(st.integers(min_value=1, max_value=64))
            channels = draw(st.integers(min_value=1, max_value=64))
            batch = draw(st.integers(min_value=1, max_value=4))
            assume(persistable == True)

        transpose_X = draw(st.booleans())
        transpose_Y = draw(st.booleans())
        len_X = draw(st.integers(min_value=1, max_value=4))
        len_Y = draw(st.integers(min_value=1, max_value=4))

        if persistable:
            assume(
                (len_X == 1 and len_Y == 1) or (len_X == 2 and len_Y == 2) or
                (len_X == 4 and len_Y == 4) or (len_X == 4 and len_Y == 2) or
                (len_X == 4 and len_Y == 1) or (len_X == 3 and len_Y == 3) or
                (len_X == 3 and len_Y == 2) or (len_X == 3 and len_Y == 1))

            if (len_X == 1 and len_Y == 1):
                X_shape = [shape0]
                Y_shape = [shape0]
                assume(transpose_X == transpose_Y)
            if (len_X == 2 and len_Y == 2):
                if ((not transpose_X) and (not transpose_Y)):
                    X_shape = [shape0, shape1]
                    Y_shape = [shape1, shape2]
                if ((transpose_X) and (not transpose_Y)):
                    X_shape = [shape1, shape0]
                    Y_shape = [shape1, shape2]
                if ((not transpose_X) and (transpose_Y)):
                    X_shape = [shape0, shape1]
                    Y_shape = [shape2, shape1]
                if ((transpose_X) and (transpose_Y)):
                    X_shape = [shape1, shape0]
                    Y_shape = [shape2, shape1]
            if (len_X == 4 and len_Y == 4):
                if ((not transpose_X) and (not transpose_Y)):
                    X_shape = [batch, channels, shape0, shape1]
                    Y_shape = [batch, channels, shape1, shape2]
                if ((transpose_X) and (not transpose_Y)):
                    X_shape = [batch, channels, shape1, shape0]
                    Y_shape = [batch, channels, shape1, shape2]
                if ((not transpose_X) and (transpose_Y)):
                    X_shape = [batch, channels, shape0, shape1]
                    Y_shape = [batch, channels, shape2, shape1]
                if ((transpose_X) and (transpose_Y)):
                    X_shape = [batch, channels, shape1, shape0]
                    Y_shape = [batch, channels, shape2, shape1]
            if (len_X == 4 and len_Y == 2):
                if ((not transpose_X) and (not transpose_Y)):
                    X_shape = [batch, channels, shape0, shape1]
                    Y_shape = [shape1, shape2]
                if ((transpose_X) and (not transpose_Y)):
                    X_shape = [batch, channels, shape1, shape0]
                    Y_shape = [shape1, shape2]
                if ((not transpose_X) and (transpose_Y)):
                    X_shape = [batch, channels, shape0, shape1]
                    Y_shape = [shape2, shape1]
                if ((transpose_X) and (transpose_Y)):
                    X_shape = [batch, channels, shape1, shape0]
                    Y_shape = [shape2, shape1]
            if (len_X == 4 and len_Y == 1):
                assume(transpose_X == transpose_Y == False)
                X_shape = [batch, channels, shape0, shape1]
                Y_shape = [shape1]
            if (len_X == 3 and len_Y == 3):
                if ((not transpose_X) and (not transpose_Y)):
                    X_shape = [channels, shape0, shape1]
                    Y_shape = [channels, shape1, shape2]
                if ((transpose_X) and (not transpose_Y)):
                    X_shape = [channels, shape1, shape0]
                    Y_shape = [channels, shape1, shape2]
                if ((not transpose_X) and (transpose_Y)):
                    X_shape = [channels, shape0, shape1]
                    Y_shape = [channels, shape2, shape1]
                if ((transpose_X) and (transpose_Y)):
                    X_shape = [channels, shape1, shape0]
                    Y_shape = [channels, shape2, shape1]
            if (len_X == 3 and len_Y == 2):
                if ((not transpose_X) and (not transpose_Y)):
                    X_shape = [channels, shape0, shape1]
                    Y_shape = [shape1, shape2]
                if ((transpose_X) and (not transpose_Y)):
                    X_shape = [channels, shape1, shape0]
                    Y_shape = [shape1, shape2]
                if ((not transpose_X) and (transpose_Y)):
                    X_shape = [channels, shape0, shape1]
                    Y_shape = [shape2, shape1]
                if ((transpose_X) and (transpose_Y)):
                    X_shape = [channels, shape1, shape0]
                    Y_shape = [shape2, shape1]
            if (len_X == 3 and len_Y == 1):
                assume(transpose_X == transpose_Y == False)
                X_shape = [channels, shape0, shape1]
                Y_shape = [shape1]
        else:
            assume((len_X == 4 and len_Y == 4))
            if ((not transpose_X) and (not transpose_Y)):
                X_shape = [batch, channels, shape0, shape1]
                Y_shape = [batch, channels, shape1, shape2]
            if ((transpose_X) and (not transpose_Y)):
                X_shape = [batch, channels, shape1, shape0]
                Y_shape = [batch, channels, shape1, shape2]
            if ((not transpose_X) and (transpose_Y)):
                X_shape = [batch, channels, shape0, shape1]
                Y_shape = [batch, channels, shape2, shape1]
            if ((transpose_X) and (transpose_Y)):
                X_shape = [batch, channels, shape1, shape0]
                Y_shape = [batch, channels, shape2, shape1]

        matmul_v2_op = OpConfig(
            type="matmul_v2",
            inputs={"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["output_data"]},
            attrs={"trans_x": transpose_X,
                   "trans_y": transpose_Y})
        if persistable:
            if target_str == "OpenCL":
                program_config = ProgramConfig(
                    ops=[matmul_v2_op],
                    weights={"input_data_y": TensorConfig(shape=Y_shape)},
                    inputs={"input_data_x": TensorConfig(shape=X_shape), },
                    outputs=["output_data"])
            else:
                program_config = ProgramConfig(
                    ops=[matmul_v2_op],
                    weights={},
                    inputs={
                        "input_data_x": TensorConfig(shape=X_shape),
                        "input_data_y": TensorConfig(shape=Y_shape)
                    },
                    outputs={"output_data"})
        else:
            program_config = ProgramConfig(
                ops=[matmul_v2_op],
                weights={},
                inputs={
                    "input_data_x": TensorConfig(shape=X_shape),
                    "input_data_y": TensorConfig(shape=Y_shape)
                },
                outputs={"output_data"})
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        if target_str == "Metal":
            atol, rtol = 1e-1, 1e-1

        return self.get_predictor_configs(), ["matmul_v2"], (atol, rtol)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            if target_type == TargetType.Metal:
                in_shape = list(program_config.inputs["input_data_x"].shape)
                if len(in_shape) != 4:
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support this op in a specific case on metal. We need to fix it as soon as possible."
        )

        def _teller2(program_config, predictor_config):
            x_shape = list(program_config.inputs["input_data_x"].shape)
            transpose_X = program_config.ops[0].attrs["trans_x"]
            if ((predictor_config.target() == TargetType.ARM) or
                (predictor_config.target() == TargetType.X86)):
                y_shape = list(program_config.inputs["input_data_y"].shape)
                if len(x_shape) == 1 and len(
                        y_shape) == 1 and transpose_X == True:
                    return True

        self.add_ignore_check_case(
            _teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support this op in a specific case on arm when x_dims=1 and x_trans=true. We need to fix it as soon as possible."
        )

        def _teller3(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "nvidia_tensorrt":
                x_shape = program_config.inputs["input_data_x"].shape
                y_shape = program_config.inputs["input_data_y"].shape
                if len(x_shape) < 3 or len(x_shape) != len(y_shape):
                    return True

        self.add_ignore_check_case(
            _teller3, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'x_shape_size != y_shape_size' or 'x_shape_size < 3' on nvidia_tensorrt."
        )

    def test(self, *args, **kwargs):
        sample_size = 100
        target_str = self.get_target()
        if target_str == "OpenCL":
            sample_size = 100
        elif target_str == "Metal":
            sample_size = 200
        self.run_and_statis(quant=False, max_examples=sample_size)


if __name__ == "__main__":
    unittest.main(argv=[''])
