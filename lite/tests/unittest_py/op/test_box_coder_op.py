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
import argparse
from functools import partial
import random
import numpy as np


class TestBoxCoderOp(AutoScanTest):
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
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

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
        num_pri = draw(st.integers(min_value=10, max_value=100))
        priorbox_shape = [num_pri, 4]
        code_type = draw(
            st.sampled_from(["encode_center_size", "decode_center_size"]))
        axis = draw(st.sampled_from([0, 1]))
        box_normalized = draw(st.booleans())
        variance = draw(st.sampled_from([[0.1, 0.2, 0.3, 0.4], []]))
        lod_data = [[1, 1, 1, 1, 1]]

        if code_type == "encode_center_size":
            targetbox_shape = draw(st.sampled_from([[30, 4], [80, 4]]))
        else:
            num0 = 1
            num1 = 1
            num2 = 1
            if axis == 0:
                num1 = priorbox_shape[0]
                num0 = np.random.randint(1, 100)
            else:
                num0 = priorbox_shape[0]
                num1 = np.random.randint(1, 100)
            num2 = priorbox_shape[1]
            targetbox_shape = draw(st.sampled_from([[num0, num1, num2]]))

        def generate_priorbox(*args, **kwargs):
            return np.random.random(priorbox_shape).astype(np.float32)

        def generate_priorbox_var(*args, **kwargs):
            return np.random.random(priorbox_shape).astype(np.float32)

        def generate_targetbox(*args, **kwargs):
            return np.random.random(targetbox_shape).astype(np.float32)

        input_type_dict = {}
        input_data_dict = {}
        input_type_dict["PriorBox"] = ["priorbox_data"]
        input_type_dict["TargetBox"] = ["targetbox_data"]
        input_data_dict["priorbox_data"] = TensorConfig(
            data_gen=partial(generate_priorbox), lod=lod_data)
        input_data_dict["targetbox_data"] = TensorConfig(
            data_gen=partial(generate_targetbox), lod=lod_data)
        if len(variance) == 0:
            input_type_dict["PriorBoxVar"] = ["priorbox_var_data"]
            input_data_dict["priorbox_var_data"] = TensorConfig(
                data_gen=partial(generate_priorbox_var))

        box_coder_op = OpConfig(
            type="box_coder",
            inputs=input_type_dict,
            outputs={"OutputBox": ["outputbox_data"]},
            attrs={
                "code_type": code_type,
                "box_normalized": box_normalized,
                "axis": axis,
                "variance": variance
            })

        program_config = ProgramConfig(
            ops=[box_coder_op],
            weights={},
            inputs=input_data_dict,
            outputs=["outputbox_data"])
        return program_config

    def sample_predictor_configs(self):
        # code_type = "encode_center_size", abs_error = 1e-4. out = out /variance
        # code_type = "decode_center_size", abs_error=1e-5.
        return self.get_predictor_configs(), ["box_coder"], (1e-4, 2e-4)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            # fp32 and fp16 will have 30% diff when data is small(1e-4), so it is not suitable to be denominator
            # in type "encode_center_size", we skip it.
            if predictor_config.target() == TargetType.ARM:
                if predictor_config.precision() == PrecisionType.FP16:
                    if program_config.ops[0].attrs[
                            "code_type"] == "encode_center_size":
                        return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.ACCURACY_ERROR,
            "Lite has little diff in a specific case on arm fp16")

    def test(self, *args, **kwargs):
        target_str = self.get_target()
        max_examples = 100
        if target_str == "OpenCL":
            # Make sure to generate enough valid cases for OpenCL
            max_examples = 600

        self.run_and_statis(
            quant=False, min_success_num=25, max_examples=max_examples)


if __name__ == "__main__":
    unittest.main(argv=[''])
