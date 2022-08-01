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
from hypothesis import given, settings, seed, example, assume, reproduce_failure
import hypothesis.strategies as st
import numpy as np
from functools import partial


class TestDistributeFpnProposalsOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.Host,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        def gen_lod_info_data(image_num, max_roi_num_per_image):
            lod_info_data_init_list = [
                np.random.randint(1, max_roi_num_per_image + 1)
                for i in range(image_num)
            ]
            lod_info_data_init_list.insert(0, 0)
            lod_info_data_init_array = np.array(lod_info_data_init_list)
            lod_info_data = np.cumsum(lod_info_data_init_array)
            return [lod_info_data.tolist()]

        image_num = draw(st.integers(min_value=1, max_value=10))
        max_roi_num_per_image = draw(st.integers(min_value=1, max_value=20))
        lod_info_data = gen_lod_info_data(image_num, max_roi_num_per_image)

        min_level = draw(st.integers(min_value=1, max_value=10))
        max_level = draw(st.integers(min_value=5, max_value=15))
        assume(max_level > min_level)
        refer_level = draw(st.integers())
        assume(refer_level <= max_level)
        assume(refer_level >= min_level)
        refer_scale = draw(st.integers(min_value=1, max_value=1000))
        pixel_offset = False  #draw(st.booleans())

        input_fpn_rois_dtype = draw(st.sampled_from(["float32"]))

        def gen_input_fpn_rois(*args, **kwargs):
            image_shape = [512, 512]
            total_rois_num = lod_info_data[-1][-1]
            fpn_rois = []
            for i in range(total_rois_num):
                xywh = np.random.rand(4)
                xy1 = xywh[0:2] * 20
                wh = xywh[2:4] * (image_shape - xy1)
                xy2 = xy1 + wh
                roi = [xy1[0], xy1[1], xy2[0], xy2[1]]
                fpn_rois.append(roi)

            if kwargs['dtype'] == 'float32':
                return np.array(fpn_rois).astype(np.float32)
            elif kwargs['dtype'] == 'float64':
                return np.array(fpn_rois).astype(np.float64)

        def gen_input_rois_num():
            rois_num = np.diff(lod_info_data[-1])
            return np.array(rois_num).astype(np.int32)

        def gen_op_inputs_outputs(*args, **kwargs):
            # 1. Generate inputs variable and inputs_tensor
            inputs = {"FpnRois": ['input_data_fpn_rois']}
            inputs_tensor = {
                'input_data_fpn_rois': TensorConfig(
                    data_gen=partial(
                        gen_input_fpn_rois, dtype=input_fpn_rois_dtype),
                    lod=lod_info_data)
            }
            has_input_rois_num = draw(st.booleans())
            if has_input_rois_num:
                inputs['RoisNum'] = ['input_data_rois_num']
                inputs_tensor['input_data_rois_num'] = TensorConfig(
                    data_gen=partial(gen_input_rois_num))

            # 2. Generate outputs variable
            total_level_num = max_level - min_level + 1
            outputs = {
                'MultiFpnRois': [
                    'output_data_multi_fpn_rois_{}'.format(idx)
                    for idx in range(total_level_num)
                ],
                'RestoreIndex': ['output_data_restore_index']
            }
            if has_input_rois_num:
                outputs['MultiLevelRoIsNum'] = [
                    ('output_data_multi_level_rois_num_{}_level'.format(idx))
                    for idx in range(total_level_num)
                ]

            # 3. Generate outputs_tensor variable name
            outputs_tensor_var_name = []
            for output_name_list in outputs.values():
                outputs_tensor_var_name.extend(output_name_list)

            # 4. Generate outputs variable data type
            outputs_dtype = {'RestoreIndex': np.int32}
            if kwargs['multi_fpn_rois_dtype'] == 'float32':
                outputs_dtype['MultiFpnRois'] = np.float32
            elif kwargs['multi_fpn_rois_dtype'] == 'float64':
                outputs_dtype['MultiFpnRois'] = np.float64

            if 'MultiLevelRoIsNum' in outputs:
                outputs_dtype['MultiLevelRoIsNum'] = np.int32
            return inputs, inputs_tensor, outputs, outputs_tensor_var_name, outputs_dtype

        inputs, inputs_tensor, outputs, outputs_tensor_var_name, outputs_dtype = gen_op_inputs_outputs(
            multi_fpn_rois_dtype=input_fpn_rois_dtype)

        distribute_fpn_proposals_op = OpConfig(
            type="distribute_fpn_proposals",
            inputs=inputs,
            outputs=outputs,
            outputs_dtype=outputs_dtype,
            attrs={
                "min_level": min_level,
                "max_level": max_level,
                "refer_level": refer_level,
                "refer_scale": refer_scale,
                "pixel_offset": pixel_offset
            })

        program_config = ProgramConfig(
            ops=[distribute_fpn_proposals_op],
            weights={},
            inputs=inputs_tensor,
            outputs=outputs_tensor_var_name)
        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["distribute_fpn_proposals"], (
            1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
