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

from auto_scan_test import AutoScanTest, SkipReasons
from auto_scan_test_rpc import AutoScanTest as RPCAutoScanTest
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest


class TestGroupNormOpBase(AutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def add_skip_pass_case(self):
        pass

    def sample_program_configs(self, *args, **kwargs):
        def generate_input(*args, **kwargs):
            return np.random.random(kwargs['in_shape']).astype(np.float32)
        def generate_scale(*args, **kwargs):
            return np.random.random([kwargs['in_shape'][1]]).astype(np.float32) + 0.5
        def generate_bias(*args, **kwargs):
            return np.random.random([kwargs['in_shape'][1]]).astype(np.float32)

        build_ops = [
            OpConfig(
                type = "group_norm",
                inputs = {
                    "X" : ["input_data"],
                    "Scale" : ["scale_data"],
                    "Bias" : ["bias_data"]
                    },
                outputs = {
                    "Y": ["output_data"],
                    "Mean" : ["mean1"],
                    "Variance" : ["var1"]},
                attrs = {
                    "data_layout" : "NCHW",
                    "epsilon" : float(1e-5),
                    "groups" : kwargs["group"]
                }),
            #with_channel
            OpConfig(
                type = "group_norm",
                inputs = {
                    "X" : ["input_data"],
                    "Scale" : ["scale_data"],
                    "Bias" : ["bias_data"]
                    },
                outputs = {
                    "Y": ["output_data"],
                    "Mean" : ["mean1"],
                    "Variance" : ["var1"]},
                attrs = {
                    "data_layout" : "NCHW",
                    "channels" : kwargs["in_shape"][1],
                    "epsilon" : float(1e-5),
                    "groups" : kwargs["group"]
                }),
            ]
        for run_op in build_ops:
            program_config = ProgramConfig(
                ops=[run_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input, *args, **kwargs)),
                    "scale_data":
                    TensorConfig(data_gen=partial(generate_scale, *args, **kwargs)),
                    "bias_data":
                    TensorConfig(data_gen=partial(generate_bias, *args, **kwargs)),
                },
                outputs=["output_data", "mean1", "var1"])
            yield program_config

class ARMTestGroupNormOpBase(RPCAutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True

    def add_skip_pass_case(self):
        pass

    def sample_program_configs(self, *args, **kwargs):
        def generate_input(*args, **kwargs):
            return np.random.random(kwargs['in_shape']).astype(np.float32)
        def generate_scale(*args, **kwargs):
            return np.random.random([kwargs['in_shape'][1]]).astype(np.float32) + 0.5
        def generate_bias(*args, **kwargs):
            return np.random.random([kwargs['in_shape'][1]]).astype(np.float32)

        build_ops = [
            OpConfig(
                type = "group_norm",
                inputs = {
                    "X" : ["input_data"],
                    "Scale" : ["scale_data"],
                    "Bias" : ["bias_data"]
                    },
                outputs = {
                    "Y": ["output_data"],
                    "Mean" : ["mean1"],
                    "Variance" : ["var1"]},
                attrs = {
                    "data_layout" : "NCHW",
                    "epsilon" : float(1e-5),
                    "groups" : kwargs["group"]
                }),
            #with_channel
            OpConfig(
                type = "group_norm",
                inputs = {
                    "X" : ["input_data"],
                    "Scale" : ["scale_data"],
                    "Bias" : ["bias_data"]
                    },
                outputs = {
                    "Y": ["output_data"],
                    "Mean" : ["mean1"],
                    "Variance" : ["var1"]},
                attrs = {
                    "data_layout" : "NCHW",
                    "channels" : kwargs["in_shape"][1],
                    "epsilon" : float(1e-5),
                    "groups" : kwargs["group"]
                }),
            ]
        for run_op in build_ops:
            program_config = ProgramConfig(
                ops=[run_op],
                weights={},
                inputs={
                    "input_data":
                    TensorConfig(data_gen=partial(generate_input, *args, **kwargs)),
                    "scale_data":
                    TensorConfig(data_gen=partial(generate_scale, *args, **kwargs)),
                    "bias_data":
                    TensorConfig(data_gen=partial(generate_bias, *args, **kwargs)),
                },
                outputs=["output_data", "mean1", "var1"])
            yield program_config
