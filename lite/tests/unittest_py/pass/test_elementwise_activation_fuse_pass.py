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

from auto_scan_test import FusePassAutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestElementwiseActivationFuse(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.ARM, [PrecisionType.FP32], DataLayoutType.NCHW, thread=[1, 4])
        self.enable_testing_on_place(TargetType.X86, [PrecisionType.FP32], DataLayoutType.NCHW, thread=[1, 4])
        #out diff
        '''
        opencl_places = [Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageDefault),
                          Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageFolder),
                          Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageDefault),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageFolder),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
                          Place(TargetType.Host, PrecisionType.FP32)    
                        ]
        self.enable_testing_on_place(places=opencl_places)
        '''
        

    def is_program_valid(self, program_config: ProgramConfig , predictor_config: CxxConfig) -> bool:
        return  True

    def sample_program_configs(self, draw):
        
        def trim_trailing_singular_dims(dims):
            actual_dims_size = len(dims)
            i = actual_dims_size
            for i in range(actual_dims_size, 0, -1):
                if dims[i-1] != 1:
                    break
            if i == len(dims):
                return dims
            
            trim_dims = []
            for j in range(0, i):
                trim_dims.append(dims[i])
            return trim_dims
            
        elementwise_type = draw(st.sampled_from(["elementwise_add", "elementwise_sub", "elementwise_mul"]))
        in_shape_x = draw(st.lists(st.integers(min_value = 1, max_value = 20), min_size = 2, max_size = 5))
        in_shape_y = draw(st.lists(st.integers(min_value = 1, max_value = 20), min_size = 2, max_size = 5))   
        
        #infer shape
        max_dim = max(len(in_shape_x), len(in_shape_y))
        axis = draw(st.integers(min_value = -1, max_value = max_dim))
        if len(in_shape_x) == len(in_shape_y):
            assume(axis== -1 or axis == 0)
        assume(axis >= -max_dim and axis < max_dim)
        axis_ = abs(len(in_shape_x) - len(in_shape_y)) + axis + 1 if axis < 0 else  axis
        
        #GetBroadcastDimsArrays
        assume(axis_ >= 0)
        assume(axis_ < max_dim)
        x_dims_array=[]
        y_dims_array=[]
        if (len(in_shape_x) > len(in_shape_y)):
            x_dims_array = in_shape_x
            for i in range(0,axis_):
                y_dims_array.append(1)
            y_dims_array = y_dims_array + in_shape_y          
            assume(axis_ + len(in_shape_y) < max_dim)#Paddle error???
            if axis_ + len(in_shape_y) < max_dim:
                for i in range(axis_ + len(in_shape_y), max_dim):
                    y_dims_array.append(1)             
        else:
            y_dims_array = in_shape_y
            for i in range(0,axis_):
                x_dims_array.append(1)
            x_dims_array = x_dims_array + in_shape_x             
            assume(axis_ + len(in_shape_x) < max_dim)#Paddle error???
            if axis_ + len(in_shape_x) < max_dim:            
                for i in range(axis_ + len(in_shape_x), max_dim):
                    x_dims_array.append(1)
        for i in range(0, max_dim):              
            assume(x_dims_array[i] == y_dims_array[i] or x_dims_array[i] <= 1 or y_dims_array[i] <= 1)
        
        #ElementwiseComputeEx
        axis_ = abs(len(in_shape_x) - len(in_shape_y)) if axis == -1 else axis         
        assume(axis_ >= 0)          
        assume(axis_ < max_dim)

        if len(in_shape_x) > len(in_shape_y):             
            y_dims_trimed = trim_trailing_singular_dims(in_shape_y)      
            axis_trim = in_shape_x if len(y_dims_trimed) == 0 else axis_
            for i in range (len(y_dims_trimed)):
                assume(i + axis_trim < len(in_shape_x)) # Paddle error???
                if in_shape_x[i + axis_trim] != y_dims_trimed[i]:
                    assume(in_shape_x[i + axis_trim] == 1 or y_dims_trimed[i]==1)
        else:    
            x_dims_trimed = trim_trailing_singular_dims(in_shape_x)
            axis_trim = in_shape_y if len(x_dims_trimed) == 0 else axis_            
            for i in range (len(x_dims_trimed)):
                assume(i + axis_trim < len(in_shape_y)) # Paddle error???                
                if in_shape_y[i + axis_trim] != x_dims_trimed[i]:
                    assume(in_shape_y[i + axis_trim] == 1 or x_dims_trimed[i]==1)

        elementwise_op = OpConfig(
            type = elementwise_type,
            inputs = {"X": ["input_data_x"],"Y": ["input_data_y"]},
            outputs = {"Out": ["elementwise_output_data"]},
            attrs = {
                "data_format": 'nchw',
                "axis": axis
            })

        act_type = draw(st.sampled_from(['relu']))
        def generate_act_attrs(act_type_str):
            attrs = {}
            if act_type_str == 'relu':
                attrs = {}
            return attrs

        active_op = OpConfig(
            type = act_type,
            inputs = {"X": ["elementwise_output_data"]},
            outputs = {"Out": ["output_data"]},
            attrs = generate_act_attrs(act_type))

        ops = [elementwise_op, active_op]
        self.ops = ops
        program_config = ProgramConfig(
            ops = ops,
            weights = {},
            inputs = {
                "input_data_x":TensorConfig(shape = in_shape_x),
                "input_data_y":TensorConfig(shape = in_shape_y)
            },
            outputs = ["output_data"])
        return program_config
    
    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), ["fusion_" + self.ops[0].type + "_activation"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300, passes=["lite_elementwise_activation_fuse_pass"])

if __name__ == "__main__":
    unittest.main(argv=[''])
