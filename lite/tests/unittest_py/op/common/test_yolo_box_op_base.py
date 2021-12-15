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

from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import numpy as np
from functools import partial
from typing import Optional, List, Callable, Dict, Any, Set
import unittest
import hypothesis
from hypothesis import assume
import hypothesis.strategies as st


def sample_program_configs(draw):
    # in_shape= [N,C,H,W]
    # H = W
    in_shape = draw(
        st.lists(
            st.integers(
                min_value=5, max_value=100), min_size=4, max_size=4))
    in_shape[2] = in_shape[3]

    # [N,2]
    ImgSize_shape = draw(st.lists(st.integers(), min_size=2, max_size=2))
    ImgSize_shape[0] = in_shape[0]
    ImgSize_shape[1] = 2

    anchors_data = draw(
        st.lists(
            st.integers(
                min_value=10, max_value=13), min_size=4, max_size=4))

    class_num_data = draw(st.integers(min_value=2, max_value=4))
    downsample_ratio_data = draw(st.sampled_from([32, 64]))
    conf_thresh_data = draw(st.floats(min_value=0.01, max_value=0.1))
    clip_bbox_data = draw(st.booleans())
    scale_x_y_data = draw(st.floats(min_value=1.0, max_value=1.0))
    iou_aware_data = draw(st.booleans())
    iou_aware_factor_data = draw(st.floats(min_value=0, max_value=1))

    def generate_ImgSize_data():
        return np.ones(ImgSize_shape).astype(np.int32)

    anchor_size = 2  # len(anchors_data) / 2

    # in paddle : develop and v2.1.2 differs here
    if (iou_aware_data):
        in_shape[1] = anchor_size * (5 + class_num_data)
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
