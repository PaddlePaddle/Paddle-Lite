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

from auto_scan_test import FusePassAutoScanTest, SkipReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st

class TestConvActiveFusePass(FusePassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def sample_program_configs(self, *args, **kwargs):
        def generate_input(*args, **kwargs):
            return np.random.random(kwargs['in_shape']).astype(np.float32)
        
        def generate_weight(*args, **kwargs):
            return np.random.random(kwargs['weight_shape']).astype(np.float32)

        conv_config = {
            "op_type": "conv2d",
            "op_inputs": {
                "Input": ["input_data"],
                "Filter":["weight_data"]
            },
            "op_outputs": {
                "Output": ["conv_output_data"]
            },
            "op_attrs": {
                "data_format": 'NCHW',
                "dilations": kwargs["dilations"],
                "padding_algorithm": kwargs['padding_algorithm'],
                "groups": kwargs["groups"],
                "paddings": kwargs["paddings"],
                "strides": kwargs["strides"]
            }
        }
        
        active_configs = [
        {
            "op_type": "relu",
            "op_inputs": {
                "X": ["conv_output_data"],
            },
            "op_outputs": {
                "Out": ["output_data"]
            },
            "op_attrs": {
            }
        },
        {
            "op_type": "leaky_relu",
            "op_inputs": {
                "X": ["conv_output_data"],
            },
            "op_outputs": {
                "Out": ["output_data"]
            },
            "op_attrs": {
                "alpha" : kwargs["alpha"]
            }
        },
        {
            "op_type": "hard_swish",
            "op_inputs": {
                "X": ["conv_output_data"],
            },
            "op_outputs": {
                "Out": ["output_data"]
            },
            "op_attrs": {
                "threshold" : kwargs["threshold"],
                "scale" : kwargs["scale"],
                "offset" : kwargs["offset"],
            }
        }
        ]

        for active_config in active_configs:
            ops_config = [conv_config, active_config]
            ops = self.generate_op_config(ops_config)
            program_config = ProgramConfig(
                ops=ops,
                weights={
                    "weight_data":
                    TensorConfig(data_gen=partial(generate_weight, *args, **kwargs)),
                },
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input, *args, **kwargs)),
                },
                outputs=["output_data"])
            yield program_config

    def sample_predictor_configs(self, program_config):
        config = CxxConfig()
        config.set_valid_places({Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageDefault),
                                 Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageFolder), 
                                 Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
                                 Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageDefault), 
                                 Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageFolder), 
                                 Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),  
                                 Place(TargetType.X86, PrecisionType.FP32),
                                 Place(TargetType.Host, PrecisionType.FP32)})
        yield config, (1e-5, 1e-5)

    def add_skip_pass_case(self):
        pass

    @given(
        in_shape=st.lists(st.integers(min_value=1, max_value=64), min_size=4, max_size=4),
        weight_shape=st.lists(st.integers(min_value=1, max_value=64), min_size=4, max_size=4),
        paddings=st.sampled_from([[1, 2], [4, 2]]),
        dilations=st.sampled_from([[1, 1]]),
        groups=st.sampled_from([1]),
        padding_algorithm=st.sampled_from(["VALID", "SAME"]),
        strides=st.sampled_from([[1, 1], [2, 2]]),
        threshold=st.floats(min_value=0, max_value=1),
        alpha=st.floats(min_value=0, max_value=1),
        scale=st.floats(min_value=0, max_value=5),
        offset=st.floats(min_value=0, max_value=1),
        )
    def test(self, *args, **kwargs):
        assume(kwargs["in_shape"][1] == kwargs["weight_shape"][1])
        assume(kwargs["in_shape"][2] >= kwargs["weight_shape"][2])
        assume(kwargs["in_shape"][3] >= kwargs["weight_shape"][3])
        self.add_skip_pass_case()
        optimized_model = self.run_test(quant=False, *args, **kwargs)
        self.assert_op_size(4, 3, self.origin_model, optimized_model)

if __name__ == "__main__":
    unittest.main()
