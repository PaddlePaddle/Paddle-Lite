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
import hypothesis.strategies as st

def sample_program_configs(draw):
    X_shape = draw(st.sampled_from([[24, 24, 4]]))
    Y_shape = draw(st.sampled_from([[4, 24, 24]]))
    background_label = draw(st.sampled_from([1]))
    score_threshold = draw(st.sampled_from([0.1]))
    post_threshold = draw(st.sampled_from([0.1]))
    nms_top_k = draw(st.sampled_from([1]))
    keep_top_k = draw(st.sampled_from([1]))
    normalized = draw(st.booleans())
    use_gaussian = draw(st.sampled_from([0.1]))
    gaussian_sigma = draw(st.sampled_from([0.1]))
    

    matrix_nms_op = OpConfig(
        type = "matrix_nms",
        inputs = {"BBoxes" : ["input_data_BBoxes"], "Scores" : ["input_data_Scores"]},
        outputs = {"Out": ["output_data"], "Index":["output_index"], "RoisNum":["RoisNum"]},
        attrs = {"background_label":background_label, "score_threshold":score_threshold, "post_threshold":post_threshold, "nms_top_k":nms_top_k, "keep_top_k":keep_top_k, "normalized":normalized, "use_gaussian":use_gaussian, "gaussian_sigma":gaussian_sigma})

    program_config = ProgramConfig(
        ops=[matrix_nms_op],
        weights={},
        inputs={
            "input_data_BBoxes":
            TensorConfig(shape=X_shape),
            "input_data_Scores" : TensorConfig(shape=Y_shape)
        },
        outputs={"output_data"})
    return program_config
