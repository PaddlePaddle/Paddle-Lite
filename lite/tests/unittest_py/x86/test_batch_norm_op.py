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
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
import paddle.inference as paddle_infer
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st

class TestBatchNormOp(AutoScanTest):
    def is_program_valid(self, program_config: ProgramConfig) -> bool:
        return True
    
    def sample_program_configs(self, *args, **kwargs):
        def generate_input(*args, **kwargs):
            return np.random.random(kwargs['in_shape']).astype(np.float32)

        def generate_scale(*args, **kwargs):
            return np.random.random(kwargs['in_shape'][1]).astype(np.float32)

        def generate_bias(*args, **kwargs):
            return np.random.random(kwargs['in_shape'][1]).astype(np.float32)

        def generate_mean(*args, **kwargs):
            return np.random.random(kwargs['in_shape'][1]).astype(np.float32)

        def generate_var(*args, **kwargs):
            return np.random.random(kwargs['in_shape'][1]).astype(np.float32)
        
        bn_op = OpConfig(
            type = "batch_norm",
            inputs = {"X" : ["input_data"], 
                "Bias" : ["bias_data"],
                "Scale" : ["scale_data"],
                "Mean" : ["mean_data"],
                "Variance" : ["var_data"]},
            outputs = {"Y" : ["output_data"],            
                "MeanOut" : ["mean_data"],
                "VarianceOut" : ["var_data"],
                "SavedMean" : ["sm_data"],
                "SavedVariance" : ["sv_data"]},
            attrs = {"epsilon" : 1e-5,
                "momentum" : 0.9,
                "use_global_stats" : False,
                "data_layout" : "NCHW",
                "is_test" : 1,
                "trainable_statistics" : 1})
        program_config = ProgramConfig(
            ops=[bn_op],
            weights={},
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input, *args, **kwargs)),
                "bias_data":
                TensorConfig(data_gen=partial(generate_bias, *args, **kwargs)),
                "scale_data":
                TensorConfig(data_gen=partial(generate_scale, *args, **kwargs)),
                "mean_data":
                TensorConfig(data_gen=partial(generate_mean, *args, **kwargs)),
                "var_data":
                TensorConfig(data_gen=partial(generate_var, *args, **kwargs))
            },
            outputs=["output_data", "mean_data", "var_data", "sm_data", "sv_data"])
        yield program_config

    def add_skip_pass_case(self):
        pass

    def sample_predictor_configs(self, program_config):
        config = CxxConfig()
        config.set_valid_places({Place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW)})
        yield config, (1e-5, 1e-5)
    
    @given(
        in_shape = st.lists(
            st.integers(
                min_value=1, max_value=10), min_size=4, max_size=5))
    def test(self, *args, **kwargs):
        self.add_skip_pass_case()
        self.run_test(quant=False, *args, **kwargs)

if __name__ == "__main__":
    unittest.main()
