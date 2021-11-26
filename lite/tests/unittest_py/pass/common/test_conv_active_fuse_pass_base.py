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
from auto_scan_test_rpc import FusePassAutoScanTest as RPCFusePassAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st

class TestConvActiveFusePassBase(FusePassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def add_skip_pass_case(self):
        pass

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
            "op_type": "relu6",
            "op_inputs": {
                "X": ["conv_output_data"],
            },
            "op_outputs": {
                "Out": ["output_data"]
            },
            "op_attrs": {
                "threshold" : kwargs["threshold"]
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

class ARMTestConvActiveFusePass(RPCFusePassAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def add_skip_pass_case(self):
        pass

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
