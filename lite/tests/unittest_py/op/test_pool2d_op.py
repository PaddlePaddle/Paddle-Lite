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

import numpy as np
from functools import partial

class TestPool2dOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW, thread=[1,2])
        self.enable_testing_on_place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW, thread=[1,2])

    def is_program_valid(self, program_config: ProgramConfig , predictor_config: CxxConfig) -> bool:
        return True


    def sample_program_configs(self, draw):
        in_shape = draw(st.lists(st.integers(
                min_value=1, max_value=10), min_size=4, max_size=4))
        pool_type = draw(st.sampled_from(["max", "avg"]))
        padding_algorithm = draw(st.sampled_from(["SAME", "VALID"]))
        pool_padding = draw(st.sampled_from([[0, 0], [0, 0, 1, 1], [1,1,1,1],[1,1]]))
        global_pooling = draw(st.sampled_from([True, False]))
        adaptive = draw(st.sampled_from([True, False]))
        exclusive = draw(st.sampled_from([True, False]))
        ceil_mode = draw(st.sampled_from([True, False]))
        pool_stride = draw(st.sampled_from([0, 1, 2]))
        pool_size = draw(st.sampled_from([0, 1, 2]))
        pool_stride = [pool_stride,] * 2
        pool_size = [pool_size,] * 2

        # assume(len(in_shape) - len(pool_size) == 2)
        if padding_algorithm == "VALID" or padding_algorithm == "SAME":
            pool_padding = [0, 0]


        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        build_ops = OpConfig(
            type = "pool2d",
            inputs={"X": ["input_data"]},
            outputs={"Out": ["output_data"]},
            attrs={
                "pooling_type": pool_type,
                "ksize": pool_size,
                "adaptive": adaptive,
                "global_pooling": global_pooling,
                "strides": pool_stride,
                "paddings": pool_padding,
                "padding_algorithm": padding_algorithm,
                "use_cudnn": False,
                "ceil_mode": ceil_mode,
                "use_mkldnn": False,
                "exclusive": exclusive,
                "data_format": "NCHW",
            })
        program_config = ProgramConfig(
            ops=[build_ops],
            weights={},
            inputs={
                "input_data":
                TensorConfig(data_gen=partial(generate_input)),
            },
            outputs=["output_data"])
        return program_config

    # def sample_program_configs(self, draw):
    #     in_shape = draw(st.lists(st.integers(min_value=2, max_value=10), min_size=2, max_size=4))

    #     assume(len(in_shape) == 4)
    #     aligned = draw(st.sampled_from([True, False]))
    #     # box_num = draw(st.integers(min_value=1, max_value=10))
    #     box_num = in_shape[0]
    #     assume(box_num > 0)
    #     box_shape = [box_num, 4]
    #     spatial_scale = draw(st.sampled_from([0.5, 1.0, 0.25]))
    #     sampling_ratio = draw(st.sampled_from([-1]))
    #     pooled_height = in_shape[2]
    #     pooled_width = in_shape[3]

    #     def generate_input(*args, **kwargs):
    #         return np.random.random(kwargs['tensor_shape']).astype(np.float32)

    #     def generate_rois(*args, **kwargs):
    #         data = np.ones(kwargs['tensor_shape']).astype(np.float32)
    #         for i in range(0, box_num):
    #             data[i][0] = np.random.randint(1, in_shape[3])
    #             data[i][1] = np.random.randint(1, in_shape[2])
    #             data[i][2] = data[i][0] + 1
    #             data[i][3] = data[i][1] + 1
    #         return data

    #     def generate_roi_num(*args, **kwargs):
    #         # print(box_num)
    #         # return np.array([int(box_num)]).astype(np.int32);
    #         return np.random.random([box_num]).astype(np.int32)

    #     build_op = OpConfig(
    #         type = "pool2d",
    #         inputs = {
    #             "X" : ["input"],
    #             "ROIs" : ["rois"],
    #             "RoisNum" : ["roi_num"],
    #         },
    #         outputs = {"Out": ["output_data"]},
    #         attrs = {
    #             "pooled_height": pooled_height,
    #             "pooled_width": pooled_width,
    #             "spatial_scale": spatial_scale,
    #             "sampling_ratio": sampling_ratio,
    #             "aligned": aligned,
    #         })
    #     program_config = ProgramConfig(
    #         ops=[build_op],
    #         weights={},
    #         inputs={
    #             "input":
    #             TensorConfig(data_gen=partial(generate_input, tensor_shape = in_shape)),
    #             "rois":
    #             TensorConfig(data_gen=partial(generate_rois, tensor_shape = box_shape)),
    #             "roi_num":
    #             TensorConfig(data_gen=partial(generate_roi_num)),
    #         },
    #         outputs=["output_data"])
    #     return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["pool2d"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)

if __name__ == "__main__":
    unittest.main(argv=[''])
