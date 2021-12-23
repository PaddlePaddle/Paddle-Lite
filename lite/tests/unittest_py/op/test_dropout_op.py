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

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        input_data_x_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=4, max_size=4))
        dropout_prob = draw(st.floats(min_value=0, max_value=1))
        # seed is useless on inference, so fix this var to default value
        seed = 0
        dropout_implementation = draw(
            st.sampled_from(['downgrade_in_infer', 'upscale_in_train']))
        is_test = True  # is_test is fixed to True for inference
        fix_seed = True  # Setting this flag to true in unittest
        has_seed_tensor = draw(st.booleans())

        def gen_input_data_seed():
            return np.array([seed]).astype(np.int32)

        def GenOpInputs():
            inputs = {"X": ["input_data_x"]}
            inputs_tensor = {
                "input_data_x": TensorConfig(shape=input_data_x_shape)
            }
            if has_seed_tensor:
                inputs["Seed"] = ["input_data_seed"]
                inputs_tensor["input_data_seed"] = TensorConfig(
                    data_gen=partial(gen_input_data_seed))
            return inputs, inputs_tensor

        def GenOpOutputs():
            outputs = {"Out": ["output_data"]}
            outputs_tensor_list = ["output_data"]
            if not is_test:
                outputs["Mask"] = ["mask_data"]
                outputs_tensor_list.append("mask_data")
            return outputs, outputs_tensor_list

        inputs, inputs_tensor = GenOpInputs()
        outputs, outputs_tensor_list = GenOpOutputs()
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
            weights={},
            inputs=inputs_tensor,
            outputs=outputs_tensor_list)
        return program_config

    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), ["dropout"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def skip_seed_input_case(program_config, predictor_config):
            if "Seed" in program_config.ops[0].inputs.keys():
                return True
            else:
                return False

        self.add_ignore_check_case(
            skip_seed_input_case, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Paddle-Lite not support 'Seed' as the input of dropout!")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
