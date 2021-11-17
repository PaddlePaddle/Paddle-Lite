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
        
        def generate_size1(*args, **kwargs):
            return np.array([int(kwargs['in_shape'][2] * kwargs['in_scale'])]).astype(np.int32)

        def generate_size2(*args, **kwargs):
            return np.array([int(kwargs['in_shape'][3] * kwargs['in_scale'])]).astype(np.int32)

        def generate_sd(*args, **kwargs):
            return np.array([kwargs['in_scale']]).astype(np.float32)
        
        def generate_os(*args, **kwargs):
            return np.array([int(kwargs['in_shape'][2] * kwargs['in_scale']), 
                int(kwargs['in_shape'][3] * kwargs['in_scale'])]).astype(np.int32)

        interp_op = [
            #SizeTensor
            OpConfig(
                type = "nearest_interp",
                inputs = {"X" : ["input_data"], 
                    "SizeTensor" : ["size_data1", "size_data2"]},
                outputs = {"Out" : ["output_data"]},
                attrs = {"scale" : kwargs['in_scale'],
                    "out_h" : int(kwargs['in_shape'][2] * kwargs['in_scale']),
                    "out_w" : int(kwargs['in_shape'][3] * kwargs['in_scale']),
                    "align_corners" : kwargs['align_cn'],
                    "align_mode" : kwargs['align_md'],
                    "interp_method" : "nearest",
                    "data_layout" : "NCHW"}),
            OpConfig(
                type = "bilinear_interp",
                inputs = {"X" : ["input_data"], 
                    "SizeTensor" : ["size_data1", "size_data2"]},
                outputs = {"Out" : ["output_data"]},
                attrs = {"scale" : kwargs['in_scale'],
                    "out_h" : int(kwargs['in_shape'][2] * kwargs['in_scale']),
                    "out_w" : int(kwargs['in_shape'][3] * kwargs['in_scale']),
                    "align_corners" : kwargs['align_cn'],
                    "align_mode" : kwargs['align_md'],
                    "interp_method" : "bilinear",
                    "data_layout" : "NCHW"}),
            OpConfig(
                type = "nearest_interp_v2",
                inputs = {"X" : ["input_data"], 
                    "SizeTensor" : ["size_data1", "size_data2"]},
                outputs = {"Out" : ["output_data"]},
                attrs = {"scale" : [kwargs['in_scale'], kwargs['in_scale']],
                    "out_h" : int(kwargs['in_shape'][2] * kwargs['in_scale']),
                    "out_w" : int(kwargs['in_shape'][3] * kwargs['in_scale']),
                    "align_corners" : kwargs['align_cn'],
                    "align_mode" : kwargs['align_md'],
                    "interp_method" : "nearest",
                    "data_layout" : "NCHW"}),
            OpConfig(
                type = "bilinear_interp_v2",
                inputs = {"X" : ["input_data"], 
                    "SizeTensor" : ["size_data1", "size_data2"]},
                outputs = {"Out" : ["output_data"]},
                attrs = {"scale" : [kwargs['in_scale'], kwargs['in_scale']],
                    "out_h" : int(kwargs['in_shape'][2] * kwargs['in_scale']),
                    "out_w" : int(kwargs['in_shape'][3] * kwargs['in_scale']),
                    "align_corners" : kwargs['align_cn'],
                    "align_mode" : kwargs['align_md'],
                    "interp_method" : "bilinear",
                    "data_layout" : "NCHW"}),
            #Scale
            OpConfig(
                type = "bilinear_interp",
                inputs = {"X" : ["input_data"], 
                    "Scale" : ["in_scale_data"]},
                outputs = {"Out" : ["output_data"]},
                attrs = {"scale" : kwargs['in_scale'],
                    "out_h" : int(kwargs['in_shape'][2] * kwargs['in_scale']),
                    "out_w" : int(kwargs['in_shape'][3] * kwargs['in_scale']),
                    "align_corners" : kwargs['align_cn'],
                    "align_mode" : kwargs['align_md'],
                    "interp_method" : "bilinear",
                    "data_layout" : "NCHW"}),
            OpConfig(
                type = "bilinear_interp_v2",
                inputs = {"X" : ["input_data"], 
                    "Scale" : ["in_scale_data"]},
                outputs = {"Out" : ["output_data"]},
                attrs = {"scale" : [kwargs['in_scale'], kwargs['in_scale']],
                    "out_h" : int(kwargs['in_shape'][2] * kwargs['in_scale']),
                    "out_w" : int(kwargs['in_shape'][3] * kwargs['in_scale']),
                    "align_corners" : kwargs['align_cn'],
                    "align_mode" : kwargs['align_md'],
                    "interp_method" : "bilinear",
                    "data_layout" : "NCHW"}),
            OpConfig(
                type = "nearest_interp",
                inputs = {"X" : ["input_data"], 
                    "Scale" : ["in_scale_data"]},
                outputs = {"Out" : ["output_data"]},
                attrs = {"scale" : kwargs['in_scale'],
                    "out_h" : int(kwargs['in_shape'][2] * kwargs['in_scale']),
                    "out_w" : int(kwargs['in_shape'][3] * kwargs['in_scale']),
                    "align_corners" : kwargs['align_cn'],
                    "align_mode" : kwargs['align_md'],
                    "interp_method" : "nearest",
                    "data_layout" : "NCHW"}),
            OpConfig(
                type = "nearest_interp_v2",
                inputs = {"X" : ["input_data"], 
                    "Scale" : ["in_scale_data"]},
                outputs = {"Out" : ["output_data"]},
                attrs = {"scale" : [kwargs['in_scale'], kwargs['in_scale']],
                    "out_h" : int(kwargs['in_shape'][2] * kwargs['in_scale']),
                    "out_w" : int(kwargs['in_shape'][3] * kwargs['in_scale']),
                    "align_corners" : kwargs['align_cn'],
                    "align_mode" : kwargs['align_md'],
                    "interp_method" : "nearest",
                    "data_layout" : "NCHW"}),
            #OutSize
            OpConfig(
                type = "bilinear_interp",
                inputs = {"X" : ["input_data"], 
                    "OutSize" : ["out_size"]},
                outputs = {"Out" : ["output_data"]},
                attrs = {"scale" : kwargs['in_scale'],
                    "out_h" : int(kwargs['in_shape'][2] * kwargs['in_scale']),
                    "out_w" : int(kwargs['in_shape'][3] * kwargs['in_scale']),
                    "align_corners" : kwargs['align_cn'],
                    "align_mode" : kwargs['align_md'],
                    "interp_method" : "bilinear",
                    "data_layout" : "NCHW"}),
            OpConfig(
                type = "bilinear_interp_v2",
                inputs = {"X" : ["input_data"], 
                    "OutSize" : ["out_size"]},
                outputs = {"Out" : ["output_data"]},
                attrs = {"scale" : [kwargs['in_scale'], kwargs['in_scale']],
                    "out_h" : int(kwargs['in_shape'][2] * kwargs['in_scale']),
                    "out_w" : int(kwargs['in_shape'][3] * kwargs['in_scale']),
                    "align_corners" : kwargs['align_cn'],
                    "align_mode" : kwargs['align_md'],
                    "interp_method" : "bilinear",
                    "data_layout" : "NCHW"}),
            OpConfig(
                type = "nearest_interp",
                inputs = {"X" : ["input_data"], 
                    "OutSize" : ["out_size"]},
                outputs = {"Out" : ["output_data"]},
                attrs = {"scale" : kwargs['in_scale'],
                    "out_h" : int(kwargs['in_shape'][2] * kwargs['in_scale']),
                    "out_w" : int(kwargs['in_shape'][3] * kwargs['in_scale']),
                    "align_corners" : kwargs['align_cn'],
                    "align_mode" : kwargs['align_md'],
                    "interp_method" : "nearest",
                    "data_layout" : "NCHW"}),
            OpConfig(
                type = "nearest_interp_v2",
                inputs = {"X" : ["input_data"], 
                    "OutSize" : ["out_size"]},
                outputs = {"Out" : ["output_data"]},
                attrs = {"scale" : [kwargs['in_scale'], kwargs['in_scale']],
                    "out_h" : int(kwargs['in_shape'][2] * kwargs['in_scale']),
                    "out_w" : int(kwargs['in_shape'][3] * kwargs['in_scale']),
                    "align_corners" : kwargs['align_cn'],
                    "align_mode" : kwargs['align_md'],
                    "interp_method" : "nearest",
                    "data_layout" : "NCHW"})
        ]
        
        for interp_op_each in interp_op:
            program_config = ProgramConfig(
                ops=[interp_op_each],
                weights={},
                inputs={
                    "input_data" :
                    TensorConfig(data_gen=partial(generate_input, *args, **kwargs)),
                    "size_data1" : TensorConfig(data_gen=partial(generate_size1, *args, **kwargs)),
                    "size_data2" : TensorConfig(data_gen=partial(generate_size2, *args, **kwargs)),
                    "in_scale_data" : TensorConfig(data_gen=partial(generate_sd, *args, **kwargs)),
                    "out_size" : TensorConfig(data_gen=partial(generate_os, *args, **kwargs))
                },
                outputs=["output_data"])
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
                min_value=10, max_value=20), min_size=4, max_size=4),
        in_scale = st.floats(min_value=0.3, max_value=3),
        align_cn = st.sampled_from([False, True]),
        align_md = st.sampled_from([0, 1])
    )
    def test(self, *args, **kwargs):
        self.add_skip_pass_case()
        self.run_test(quant=False, *args, **kwargs)

if __name__ == "__main__":
    unittest.main()
