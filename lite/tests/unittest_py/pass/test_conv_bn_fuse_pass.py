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
from test_conv_util import UpdatePaddingAndDilation, ConvOutputSize, ConvTransposeOutputSize
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st


class TestConvBnFuse(FusePassAutoScanTest):
    def __init__(self, *args, **kwargs):
        FusePassAutoScanTest.__init__(self, *args, **kwargs)
        self.ops = []
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.X86, [PrecisionType.FP32],
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
        #Metal not support conv2d_transpose: cannot find the name
        '''
        metal_places = [
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.Metal, PrecisionType.FP16,
                  DataLayoutType.MetalTexture2DArray)
        ]
        self.enable_testing_on_place(places=metal_places)
        '''

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        result = True
        if predictor_config.target() == TargetType.ARM:
            result = result and predictor_config.precision(
            ) != PrecisionType.FP16 and predictor_config.precision(
            ) != PrecisionType.INT8
        return result

    def sample_program_configs(self, draw):
        #conv or conv_transpose
        Transpose = draw(st.sampled_from([True, False]))

        #conv param or conv_transpose param
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=2, max_value=128),
                min_size=3,
                max_size=3))
        in_shape = [draw(st.integers(min_value=1, max_value=4))] + in_shape
        weight_shape = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=8), min_size=4, max_size=4))
        paddings = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=2), min_size=2, max_size=2))
        dilations = draw(st.sampled_from([[1, 1], [2, 2]]))
        groups = draw(st.sampled_from([1, 2, in_shape[1]]))
        padding_algorithm = draw(st.sampled_from(["VALID", "SAME"]))
        strides = draw(st.sampled_from([[1, 1], [2, 2]]))
        output_padding = draw(
            st.sampled_from([[], [
                draw(
                    st.integers(
                        min_value=0,
                        max_value=max(strides[0], dilations[0]) - 1)), draw(
                            st.integers(
                                min_value=0,
                                max_value=max(strides[1], dilations[1]) - 1))
            ]]))
        scale_in = draw(st.floats(min_value=0.001, max_value=0.1))
        scale_out = draw(st.floats(min_value=0.001, max_value=0.1))

        #active param
        threshold = draw(st.floats(min_value=0, max_value=1))
        alpha = draw(st.floats(min_value=0, max_value=1))
        scale = draw(st.floats(min_value=0.5, max_value=5))
        offset = draw(st.floats(min_value=0, max_value=1))
        slope = draw(st.floats(min_value=0.7, max_value=0.9))

        conv_out_shape = []
        paddings_, dilations_ = UpdatePaddingAndDilation(
            in_shape, weight_shape, paddings, dilations, groups,
            padding_algorithm, strides)
        self.depthwise = False

        if Transpose:
            assume(in_shape[1] == weight_shape[0])
            assume(in_shape[1] % groups == 0)  #TODO
            if len(output_padding):
                assume(output_padding[0] < max(strides[0], dilations_[0]))
                assume(output_padding[1] < max(strides[1], dilations_[1]))
            conv_out_shape = [in_shape[0], weight_shape[1] * groups]
            oh, ow = ConvTransposeOutputSize(in_shape, weight_shape,
                                             dilations_, paddings_, strides)
            if len(output_padding):
                oh = oh + output_padding[0]
                ow = ow + output_padding[1]
            conv_out_shape = conv_out_shape + [int(oh), int(ow)]
            assume(oh > 0 and ow > 0)
            if len(output_padding):
                conv_output_h = (oh + output_padding[0] + paddings[0] +
                                 paddings[1] -
                                 (dilations[0] *
                                  (weight_shape[2] - 1) + 1)) / strides[0] + 1
                conv_output_w = (oh + output_padding[1] + paddings[0] +
                                 paddings[1] -
                                 (dilations[1] *
                                  (weight_shape[3] - 1) + 1)) / strides[1] + 1
                assume(in_shape[2] == conv_output_h)
                assume(in_shape[3] == conv_output_w)

        else:
            self.depthwise = in_shape[1] == weight_shape[1] and in_shape[
                1] == groups
            assume(in_shape[1] == weight_shape[1] * groups)
            assume(weight_shape[0] % groups == 0)
            conv_out_shape = [in_shape[0], weight_shape[0]]
            oh, ow = ConvOutputSize(in_shape, weight_shape, dilations_,
                                    paddings_, strides)
            conv_out_shape = conv_out_shape + [int(oh), int(ow)]
            assume(oh > 0 and ow > 0)

        #bn param
        epsilon = draw(st.floats(min_value=0.00001, max_value=0.001))
        momentum = draw(st.floats(min_value=0.1, max_value=0.9))

        def generate_scale(*args, **kwargs):
            return np.random.random(
                [conv_out_shape[1]]).astype(np.float32) + 0.5

        def generate_bias(*args, **kwargs):
            return np.random.random([conv_out_shape[1]]).astype(np.float32)

        def generate_mean(*args, **kwargs):
            return np.random.random([conv_out_shape[1]]).astype(np.float32)

        def generate_variance(*args, **kwargs):
            return np.random.random([conv_out_shape[1]]).astype(np.float32)

        conv_type = ""
        conv_attrs = {}
        if Transpose:
            conv_type = "conv2d_transpose"
            conv_attrs = {
                "data_format": 'nchw',
                "dilations": dilations,
                "padding_algorithm": padding_algorithm,
                "groups": groups,
                "paddings": paddings,
                "strides": strides,
                "Scale_in": scale_in,
                "Scale_out": scale_out,
                "output_size": [],
                "output_padding": output_padding
            }
        else:
            conv_type = "conv2d"
            conv_attrs = {
                "data_format": 'nchw',
                "dilations": dilations,
                "padding_algorithm": padding_algorithm,
                "groups": groups,
                "paddings": paddings,
                "strides": strides,
                "Scale_in": scale_in,
                "Scale_out": scale_out
            }

        conv_op = OpConfig(
            type=conv_type,
            inputs={"Input": ["input_data"],
                    "Filter": ["filter_data"]},
            outputs={"Output": ["conv_output_data"]},
            attrs=conv_attrs)

        bn_op = OpConfig(
            type="batch_norm",
            inputs={
                "X": ["conv_output_data"],
                "Scale": ["scale_data"],
                "Bias": ["bias_data"],
                "Mean": ["mean_data"],
                "Variance": ["variance_data"]
            },
            outputs={
                "Y": ["output_data"],
                "MeanOut": ["mean_data"],
                "VarianceOut": ["variance_data"],
                "SavedMean": ["saved_mean"],
                "SavedVariance": ["saved_variance"]
            },
            attrs={
                "is_test": True,
                "trainable_statistics": False,
                "data_layout": "NCHW",
                "use_global_stats": False,
                "epsilon": epsilon,
                "momentum": momentum
            })

        ops = [conv_op, bn_op]
        self.ops = ops
        program_config = ProgramConfig(
            ops=ops,
            weights={
                "filter_data": TensorConfig(shape=weight_shape),
                "scale_data": TensorConfig(data_gen=partial(generate_scale)),
                "bias_data": TensorConfig(data_gen=partial(generate_bias)),
                "mean_data": TensorConfig(data_gen=partial(generate_mean)),
                "variance_data":
                TensorConfig(data_gen=partial(generate_variance)),
            },
            inputs={"input_data": TensorConfig(shape=in_shape)},
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        config = CxxConfig()
        return self.get_predictor_configs(), [self.ops[0].type], (1e-4, 1e-4)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(
            quant=False, max_examples=300, passes=["lite_conv_bn_fuse_pass"])


if __name__ == "__main__":
    unittest.main(argv=[''])
