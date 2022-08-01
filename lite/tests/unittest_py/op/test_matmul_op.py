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


class TestMulOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP16,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

        # self.enable_testing_on_place(TargetType.Metal, PrecisionType.FP32,
        #                             DataLayoutType.NCHW)
        arm_valid_places = [
            Place(TargetType.ARM, PrecisionType.INT8, DataLayoutType.NCHW),
            Place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=arm_valid_places, thread=[1, 4])
        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageFolder),
            # Place(TargetType.OpenCL, PrecisionType.FP16,
            #       DataLayoutType.ImageDefault), 
            Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.OpenCL, PrecisionType.Any,
                  DataLayoutType.ImageDefault),
            Place(TargetType.OpenCL, PrecisionType.Any,
                  DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=opencl_places)
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(
            device_names=["nvidia_tensorrt", "intel_openvino"])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True  # True ci run matmul has diff

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
        assume(shape0 != shape1)
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
                assume(transpose_X == transpose_Y)
                X_shape = [shape0]
                if ((not transpose_X) and (not transpose_Y)):
                    Y_shape = [shape0]
                if ((transpose_X) and (transpose_Y)):
                    Y_shape = [shape1]
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

        alpha = draw(st.sampled_from([0.1, 1.0, 1.1, -1.5]))
        fused_reshape_X = draw(st.sampled_from([[]]))
        fused_reshape_Y = draw(st.sampled_from([[]]))
        fused_transpose_X = draw(st.sampled_from([[]]))
        fused_transpose_Y = draw(st.sampled_from([[]]))
        fused_reshape_Out = draw(st.sampled_from([[]]))
        fused_transpose_Out = draw(st.sampled_from([[]]))
        Scale_x = draw(st.floats(min_value=0.1, max_value=10.0))
        Scale_y = draw(st.floats(min_value=0.1, max_value=10.0))
        Scale_out = draw(st.floats(min_value=0.1, max_value=10.0))
        # not use for lite
        head_number = draw(st.integers(min_value=1, max_value=1))
        force_fp32_output = draw(st.booleans())

        matmul_op = OpConfig(
            type="matmul",
            inputs={"X": ["input_data_x"],
                    "Y": ["input_data_y"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "transpose_X": transpose_X,
                "transpose_Y": transpose_Y,
                "alpha": alpha,
                "fused_reshape_X": fused_reshape_X,
                "fused_reshape_Y": fused_reshape_Y,
                "fused_transpose_X": fused_transpose_X,
                "fused_transpose_Y": fused_transpose_Y,
                "fused_reshape_Out": fused_reshape_Out,
                "fused_transpose_Out": fused_transpose_Out,
                "Scale_x": Scale_x,
                "Scale_y": Scale_y,
                "Scale_out": Scale_out,
                "head_number": head_number,
                "force_fp32_output": force_fp32_output
            })
        if persistable:
            if target_str == "OpenCL":
                program_config = ProgramConfig(
                    ops=[matmul_op],
                    weights={"input_data_y": TensorConfig(shape=Y_shape)},
                    inputs={"input_data_x": TensorConfig(shape=X_shape), },
                    outputs=["output_data"])
            else:
                program_config = ProgramConfig(
                    ops=[matmul_op],
                    weights={},
                    inputs={
                        "input_data_x": TensorConfig(shape=X_shape),
                        "input_data_y": TensorConfig(shape=Y_shape)
                    },
                    outputs=["output_data"])
        else:
            program_config = ProgramConfig(
                ops=[matmul_op],
                weights={},
                inputs={
                    "input_data_x": TensorConfig(shape=X_shape),
                    "input_data_y": TensorConfig(shape=Y_shape)
                },
                outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["matmul"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "nvidia_tensorrt":
                x_shape = program_config.inputs["input_data_x"].shape
                y_shape = program_config.inputs["input_data_y"].shape
                if len(x_shape) < 3 or len(x_shape) != len(y_shape):
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'x_shape_size != y_shape_size' or 'x_shape_size < 3' on nvidia_tensorrt."
        )

    def test(self, *args, **kwargs):
        sample_size = 100
        target_str = self.get_target()
        self.run_and_statis(quant=False, max_examples=sample_size)


if __name__ == "__main__":
    unittest.main(argv=[''])
