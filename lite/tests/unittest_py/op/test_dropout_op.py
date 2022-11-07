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


class TestDropoutOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        opencl_valid_places = [
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
        self.enable_testing_on_place(places=opencl_valid_places)
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
            "nvidia_tensorrt", "intel_openvino", "kunlunxin_xtcl"
        ])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        target_type = predictor_config.target()
        in_x_shape = list(program_config.inputs["input_data_x"].shape)
        dropout_implementation = program_config.ops[0].attrs[
            "dropout_implementation"]
        if target_type == TargetType.Metal:
            if in_x_shape[0] != 1 or len(in_x_shape) != 4 \
                or dropout_implementation != 'downgrade_in_infer':
                return False
        if target_type == TargetType.NNAdapter:
            if "Seed" in program_config.ops[0].inputs:
                return False

        return True

    def sample_program_configs(self, draw):
        # x shape need meet len(x_shape) > 1 when dropout_implementation == downgrade_in_infer!
        input_data_x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=32), min_size=2, max_size=4))
        dropout_prob = draw(st.floats(min_value=0, max_value=1))
        seed = draw(st.integers(min_value=0, max_value=1024))
        dropout_implementation = draw(
            st.sampled_from(['downgrade_in_infer', 'upscale_in_train']))
        # is_test = False only used in training
        is_test = True
        fix_seed = draw(st.booleans())

        def gen_input_data_seed():
            return np.array([seed]).astype(np.int32)

        def GenOpInputs():
            inputs = {"X": ["input_data_x"]}
            inputs_tensor = {
                "input_data_x": TensorConfig(shape=input_data_x_shape)
            }
            if draw(st.booleans()):
                inputs["Seed"] = ["input_data_seed"]
                inputs_tensor["input_data_seed"] = TensorConfig(
                    data_gen=partial(gen_input_data_seed))
            return inputs, inputs_tensor

        def GenOpOutputs():
            outputs = {"Out": ["output_data"]}
            outputs_tensor = ["output_data"]
            if is_test == False:
                outputs["Mask"] = ["mask_data"]
                outputs_tensor.append("mask_data")
            return outputs, outputs_tensor

        inputs, inputs_tensor = GenOpInputs()
        outputs, outputs_tensor = GenOpOutputs()

        dropout_op = OpConfig(
            type="dropout",
            inputs=inputs,
            outputs=outputs,
            attrs={
                "dropout_prob": dropout_prob,
                "fix_seed": fix_seed,
                "seed": seed,
                "dropout_implementation": dropout_implementation,
                "is_test": is_test
            })
        program_config = ProgramConfig(
            ops=[dropout_op],
            weights={"mask_data": TensorConfig(shape=input_data_x_shape)},
            inputs=inputs_tensor,
            outputs=outputs_tensor)
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        if target_str == "Metal":
            atol, rtol = 1e-3, 1e-3

        return self.get_predictor_configs(), ["dropout"], (atol, rtol)

    def add_ignore_pass_case(self):
        pass

        def _teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            in_x_shape = list(program_config.inputs["input_data_x"].shape)
            dropout_implementation = program_config.ops[0].attrs[
                "dropout_implementation"]
            if target_type == TargetType.Metal:
                if in_x_shape[0] != 1 or len(in_x_shape) != 4 \
                    or dropout_implementation != 'downgrade_in_infer':
                    return False

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.ACCURACY_ERROR,
            "The op output has diff in a specific case on metal. We need to fix it as soon as possible."
        )

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 300
        if target_str == "Metal":
            # Make sure to generate enough valid cases for Metal
            max_examples = 3000
        self.run_and_statis(quant=False, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
