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
from hypothesis import assume

def sample_program_configs(draw):
    # inputs
    # LoD tensor need
    image_shape = [512, 512]
    image_num = draw(st.integers(min_value = 1, max_value = 10))
    max_roi_num_per_image = draw(st.integers(min_value = 1, max_value = 20))
    def gen_lod_data(dimentsions_0_size, dimentsions_1_max_size):
        sequence_length_list = [np.random.randint(1, dimentsions_1_max_size + 1) for i in range(dimentsions_0_size)]
        lod_data_init_list = sequence_length_list
        lod_data_init_list.insert(0, 0)
        lod_data_init_array = np.array(lod_data_init_list)
        lod_data = np.cumsum(lod_data_init_array)
        return [lod_data.tolist()]
    
    lod_data = gen_lod_data(image_num, max_roi_num_per_image)
    def gen_input_fpn_rois():
        total_rois_num = lod_data[-1][-1]
        rois = []
        for i in range(total_rois_num):
            xywh = np.random.rand(4)
            xy1 = xywh[0:2] * 20
            wh = xywh[2:4] * (image_shape - xy1)
            xy2 = xy1 + wh
            roi = [xy1[0], xy1[1], xy2[0], xy2[1]]
            rois.append(roi)
        rois = np.array(rois).astype(np.float32)
        print(rois.shape)
        return rois

    def gen_input_rois_num():
        rois_num = lod_data[1:]
        return np.array(rois_num).astype(np.int32)

    # attrs
    min_level = draw(st.integers(min_value = 1, max_value = 10))
    max_level = draw(st.integers(min_value = 5, max_value = 15))
    assume(max_level > min_level)
    refer_level = draw(st.integers())
    assume(refer_level <= max_level)
    assume(refer_level >= min_level)
    refer_scale = draw(st.integers(min_value = 1))
    pixel_offset = draw(st.booleans())

    def gen_op_inputs_outputs():
        inputs = {"FpnRois": ['input_data_fpn_rois']}
        inputs_tensor = {'input_data_fpn_rois': TensorConfig(data_gen=partial(gen_input_fpn_rois), lod=lod_data)}
        if draw(st.booleans()):
            inputs['RoisNum'] = ['input_data_rois_num']
            inputs_tensor['input_data_rois_num'] = TensorConfig(data_gen=partial(gen_input_rois_num))
        outputs = {'MultiFpnRois': ['output_data_multi_fpn_rois'],
                   'RestoreIndex': ['output_data_restore_index']}
        if draw(st.booleans()):
            outputs['MultiLevelRoIsNum'] = ['output_data_multi_level_rois_num']
        outputs_tensor_var_name = [name for name in outputs.values()]
        outputs_tensor_var_name = ['output_data_restore_index', 'output_data_multi_fpn_rois']
        return inputs, inputs_tensor, outputs, outputs_tensor_var_name
    inputs, inputs_tensor, outputs, outputs_tensor_var_name = gen_op_inputs_outputs()
    distribute_fpn_proposals_op = OpConfig(
        type = "distribute_fpn_proposals",
        inputs = inputs,
        outputs = outputs,
        attrs = {"min_level" : min_level,
                "max_level": max_level,
                "refer_level": refer_level,
                "refer_scale": refer_scale,
                "pixel_offset": pixel_offset})
    program_config = ProgramConfig(
        ops=[distribute_fpn_proposals_op],
        weights={},
        inputs= inputs_tensor,
        outputs=outputs_tensor_var_name)
    return program_config
