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
from functools import partial
import random
import numpy as np


class TestYoloBoxOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)

        host_places = [
            Place(TargetType.Host, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=host_places, thread=[1, 4])

        opencl_places = [
            Place(TargetType.OpenCL, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.OpenCL, PrecisionType.FP16,
                  DataLayoutType.ImageDefault), Place(
                      TargetType.OpenCL, PrecisionType.FP16,
                      DataLayoutType.ImageFolder), Place(
                          TargetType.OpenCL, PrecisionType.Any,
                          DataLayoutType.ImageDefault), Place(
                              TargetType.OpenCL, PrecisionType.Any,
                              DataLayoutType.ImageFolder),
            Place(TargetType.OpenCL, PrecisionType.Any, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=opencl_places)

        # self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        # self.enable_devices_on_nnadapter(device_names=["nvidia_tensorrt"])

        # having diff 
        # metal_places = [
        #     Place(TargetType.Metal, PrecisionType.FP32,
        #           DataLayoutType.MetalTexture2DArray),
        #     Place(TargetType.Metal, PrecisionType.FP16,
        #           DataLayoutType.MetalTexture2DArray),
        #     Place(TargetType.ARM, PrecisionType.FP32),
        #     Place(TargetType.Host, PrecisionType.FP32)
        # ]
        # self.enable_testing_on_place(places=metal_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        # in_shape= [N,C,H,W]
        # H = W
        N = draw(st.integers(min_value=1, max_value=4))
        C = draw(st.integers(min_value=1, max_value=128))
        H = draw(st.integers(min_value=1, max_value=128))
        W = draw(st.integers(min_value=1, max_value=128))
        in_shape = draw(st.sampled_from([[N, C, H, W]]))
        in_shape[2] = in_shape[3]

        # [N,2]
        ImgSize_shape = draw(st.lists(st.integers(), min_size=2, max_size=2))
        ImgSize_shape[0] = in_shape[0]
        ImgSize_shape[1] = 2

        anchors_data = draw(
            st.lists(
                st.integers(
                    min_value=10, max_value=13),
                min_size=4,
                max_size=4))

        class_num_data = draw(st.integers(min_value=2, max_value=4))
        downsample_ratio_data = draw(st.sampled_from([32, 64]))
        conf_thresh_data = draw(st.floats(min_value=0.01, max_value=0.1))
        clip_bbox_data = draw(st.booleans())
        scale_x_y_data = draw(st.floats(min_value=1.0, max_value=1.0))
        # iou_aware_data is not supported by lite
        # I default them
        iou_aware_data = draw(st.booleans())
        iou_aware_data = False

        iou_aware_factor_data = draw(st.floats(min_value=0, max_value=1))

        def generate_ImgSize_data():
            return np.ones(ImgSize_shape).astype(np.int32)

        anchor_size = 2  # len(anchors_data) / 2

        # in paddle : develop and v2.1.2 differs here
        # here consistent with develop
        if (iou_aware_data):
            in_shape[1] = anchor_size * (6 + class_num_data)
        else:
            in_shape[1] = anchor_size * (5 + class_num_data)

        yolo_box_op = OpConfig(
            type="yolo_box",
            inputs={"X": ["input_data"],
                    "ImgSize": ["ImgSize_data"]},
            outputs={"Boxes": ["Boxes_data"],
                     "Scores": ["Scores_data"]},
            attrs={
                "anchors": anchors_data,
                "class_num": class_num_data,
                "downsample_ratio": downsample_ratio_data,
                "conf_thresh": conf_thresh_data,
                "clip_bbox": clip_bbox_data,
                "scale_x_y": scale_x_y_data,
                "iou_aware": iou_aware_data,
                "iou_aware_factor": iou_aware_factor_data
            })
        program_config = ProgramConfig(
            ops=[yolo_box_op],
            weights={},
            inputs={
                "input_data": TensorConfig(shape=in_shape),
                "ImgSize_data":
                TensorConfig(data_gen=partial(generate_ImgSize_data)),
            },
            outputs=["Boxes_data", "Scores_data"])
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["yolo_box"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=25)


if __name__ == "__main__":
    unittest.main(argv=[''])
