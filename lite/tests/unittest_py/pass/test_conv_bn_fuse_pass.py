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

from auto_scan_test import AutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestConvBnFuse(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.ops = []
        self.enable_testing_on_place(TargetType.ARM, [PrecisionType.FP32], DataLayoutType.NCHW, thread=[1, 4])
        self.enable_testing_on_place(TargetType.X86, [PrecisionType.FP32], DataLayoutType.NCHW, thread=[1, 4])        
        opencl_places = [Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageDefault),
                          Place(TargetType.OpenCL, PrecisionType.FP16, DataLayoutType.ImageFolder),
                          Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageDefault),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.ImageFolder),
                          Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
                          Place(TargetType.Host, PrecisionType.FP32)    
                        ]
        self.enable_testing_on_place(places=opencl_places)

    def is_program_valid(self, program_config: ProgramConfig , predictor_config: CxxConfig) -> bool:
        return True        

    def sample_program_configs(self, draw):
        #conv param
        in_shape = draw(st.lists(st.integers(min_value = 1, max_value = 64), min_size = 4, max_size = 4))
        kw = np.random.randint(1, 9)
        kh = np.random.randint(1, 9)
        cout = np.random.randint(1, 64)
        cin = np.random.randint(1, 64)
        scale_in = draw(st.floats(min_value = 0.001, max_value = 0.1))
        scale_out = draw(st.floats(min_value = 0.001, max_value = 0.1))
        weight_shape = [cout, cin, kh, kw]
        groups = draw(st.sampled_from([1, 2, cin]))
        val = cin * groups
        assume(in_shape[1] == val)
        assume(cout%groups==0)
        assume(in_shape[2] >= weight_shape[2])
        assume(in_shape[3] >= weight_shape[3])

        paddings = draw(st.lists(st.integers(min_value = 0, max_value = 2), min_size = 2, max_size = 2))
        dilations = draw(st.sampled_from([[1, 1]]))
        padding_algorithm = draw(st.sampled_from(["VALID", "SAME"]))
        strides = draw(st.sampled_from([[1, 1], [2, 2]]))
        data_format = "NCHW"

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)
        def generate_filter(*args, **kwargs):
            return np.random.random(weight_shape).astype(np.float32)
        def generate_conv_bias(*args, **kwargs):
            return np.random.random([cout]).astype(np.float32)

        #bn param
        is_test_val = draw(st.sampled_from([True, False]))
        epsilon = draw(st.floats(min_value = 0.00001, max_value = 0.001))
        momentum = draw(st.floats(min_value = 0.1, max_value = 0.9))

        def generate_scale(*args, **kwargs):
            return np.random.random([cout]).astype(np.float32) + 0.5
        def generate_bias(*args, **kwargs):
            return np.random.random([cout]).astype(np.float32)
        def generate_mean(*args, **kwargs):
            return np.random.random([cout]).astype(np.float32)
        def generate_variance(*args, **kwargs):
            return np.random.random([cout]).astype(np.float32)

        conv_op = OpConfig(
            type = "conv2d",
            inputs = {"Input" : ["input_data"],
                     "Filter" : ["filter_data"]},
            outputs = {"Output": ["conv_output_data"]},
            attrs = {"strides" : strides,
                    "paddings" : paddings,
                    "use_mkldnn" : True,
                    "padding_algorithm" : padding_algorithm,
                    "groups" : groups,
                    "dilations" : dilations,
                    "Scale_in" : scale_in,
                    "Scale_out" : scale_out,
                    "data_format" : data_format}
        )

        bn_op = OpConfig(
            type = "batch_norm",
            inputs = {"X": ["conv_output_data"],
                      "Scale" : ["scale_data"],
                      "Bias" : ["bias_data"],
                      "Mean" : ["mean_data"],
                      "Variance" : ["variance_data"]
                      },
            outputs = {"Y": ["output_data"],
                       "MeanOut" : ["mean_data"],
                       "VarianceOut" : ["variance_data"],
                       "SavedMean" : ["saved_mean"],
                       "SavedVariance" : ["saved_variance"]
                      },
            attrs = {
                "is_test" : True,
                "trainable_statistics" : False,
                "data_layout" : "NCHW",
                "use_global_stats" : False,
                "epsilon" : epsilon,
                "momentum" : momentum
            }
        )

        ops = [conv_op, bn_op]
        self.ops = ops
        program_config = ProgramConfig(
            ops = ops,
            weights = {
                "filter_data":
                TensorConfig(data_gen = partial(generate_filter)),
                "conv_bias_data":
                TensorConfig(data_gen = partial(generate_conv_bias))
            },
            inputs = {
                "input_data":
                TensorConfig(data_gen = partial(generate_input)),
                "scale_data":
                TensorConfig(data_gen = partial(generate_scale)),
                "bias_data":
                TensorConfig(data_gen = partial(generate_bias)),
                "mean_data":
                TensorConfig(data_gen = partial(generate_mean)),
                "variance_data":
                TensorConfig(data_gen = partial(generate_variance)),
            },
            outputs=["output_data", "mean_data", "variance_data", "saved_mean", "saved_variance"])

        return program_config
    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), [self.ops[0].type], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            return True    
        self.add_ignore_check_case(
            # IgnoreReasonsBase.PADDLE_NOT_IMPLEMENTED
            # IgnoreReasonsBase.PADDLELITE_NOT_SUPPORT
            # IgnoreReasonsBase.ACCURACY_ERROR
            teller1, IgnoreReasons.ACCURACY_ERROR,
            "conv bn fuse only one output"
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)

if __name__ == "__main__":
    unittest.main(argv=[''])
