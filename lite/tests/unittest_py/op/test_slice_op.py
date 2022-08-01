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
from functools import partial
import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st
import numpy as np


class TestSliceOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM, [PrecisionType.FP32],
            DataLayoutType.NCHW,
            thread=[1, 4])
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
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=[
            "cambricon_mlu", "nvidia_tensorrt", "intel_openvino"
        ])
        '''
        #All of metal inputs error.
        metal_places = [
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.Metal, PrecisionType.FP16,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.ARM, PrecisionType.FP32),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=metal_places)
        '''

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        # check config
        x_dtype = program_config.inputs["input_data"].dtype
        if x_dtype == np.int32 or x_dtype == np.int64:
            return False
        return True

    def sample_program_configs(self, draw):
        in_shape = draw(
            st.lists(
                st.integers(
                    min_value=6, max_value=64), min_size=4, max_size=4))
        axes = draw(st.sampled_from([[3], [0, 1], [0, 1, 2], [0, 1, 2, 3]]))
        starts = draw(st.sampled_from([[-1], [0, 1], [0, 1, 2], [0, 1, 2, 3]]))
        ends = draw(
            st.sampled_from([[10000], [1, 2], [1, 2, 3], [1, 2, 3, 4]]))
        decrease_axis = draw(
            st.sampled_from([[3], [0, 1], [0, 1, 2], [0, 1, 2, 3]]))
        infer_flags = draw(st.sampled_from([[1, 1, 1]]))
        input_num = draw(st.sampled_from([0, 1, 2]))
        input_type = draw(st.sampled_from(["float32", "int32", "int64"]))

        assume((len(starts) == len(ends)) & (len(starts) == len(axes)))
        assume(len(decrease_axis) == len(starts))
        assume(len(axes) <= len(in_shape))
        for i in range(len(starts)):
            start = starts[i] if starts[i] >= 0 else starts[i] + in_shape[axes[
                i]]
            assume(start < in_shape[axes[i]])

        if input_num == 0:
            assume(len(axes) == 2)

        def generate_input(*args, **kwargs):
            if input_type == "float32":
                return np.random.normal(0.0, 1.0, in_shape).astype(np.float32)
            elif input_type == "int32":
                return np.random.normal(0.0, 1.0, in_shape).astype(np.int32)
            elif input_type == "int64":
                return np.random.normal(0.0, 1.0, in_shape).astype(np.int64)

        def generate_starts(*args, **kwargs):
            return np.array(starts, dtype="int32")

        def generate_ends(*args, **kwargs):
            return np.array(ends, dtype="int32")

        def generate_startlist1(*args, **kwargs):
            return np.array([1], dtype="int32")

        def generate_startlist2(*args, **kwargs):
            return np.array([1], dtype="int32")

        def generate_endlist1(*args, **kwargs):
            return np.array([2], dtype="int32")

        def generate_endlist2(*args, **kwargs):
            return np.array([2], dtype="int32")

        dics_intput = [{
            "Input": ["input_data"],
            "StartsTensorList": ["StartsTensorList1", "StartsTensorList2"],
            "EndsTensorList": ["EndsTensorList1", "EndsTensorList2"]
        }, {
            "Input": ["input_data"],
            "StartsTensor": ["starts_data"],
            "EndsTensor": ["ends_data"],
            "StartsTensorList": ["StartsTensorList1", "StartsTensorList2"],
            "EndsTensorList": ["EndsTensorList1", "EndsTensorList2"]
        }, {
            "Input": ["input_data"]
        }, {}]

        dics_weight = [{
            "StartsTensorList1":
            TensorConfig(data_gen=partial(generate_startlist1)),
            "StartsTensorList2":
            TensorConfig(data_gen=partial(generate_startlist2)),
            "EndsTensorList1":
            TensorConfig(data_gen=partial(generate_endlist1)),
            "EndsTensorList2":
            TensorConfig(data_gen=partial(generate_endlist2))
        }, {
            "starts_data": TensorConfig(data_gen=partial(generate_starts)),
            "ends_data": TensorConfig(data_gen=partial(generate_ends)),
            "StartsTensorList1":
            TensorConfig(data_gen=partial(generate_startlist1)),
            "StartsTensorList2":
            TensorConfig(data_gen=partial(generate_startlist2)),
            "EndsTensorList1":
            TensorConfig(data_gen=partial(generate_endlist1)),
            "EndsTensorList2":
            TensorConfig(data_gen=partial(generate_endlist2))
        }, {}]

        ops_config = OpConfig(
            type="slice",
            inputs=dics_intput[input_num],
            outputs={"Out": ["output_data"]},
            attrs={
                "axes": axes,
                "starts": starts,
                "ends": ends,
                "decrease_axis": decrease_axis,
                "infer_flags": infer_flags,
                "input_num": input_num
            })

        program_config = ProgramConfig(
            ops=[ops_config],
            weights=dics_weight[input_num],
            inputs={
                "input_data": TensorConfig(data_gen=partial(generate_input))
            },
            outputs=["output_data"])

        return program_config

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), ["slice"], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        def teller1(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "nvidia_tensorrt":
                input_num = len(program_config.ops[0].inputs)
                in_shape = program_config.inputs["input_data"].shape
                axes = program_config.ops[0].attrs["axes"]
                if input_num != 1 or len(in_shape) == 1 or 0 in axes:
                    return True

        self.add_ignore_check_case(
            teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support 'input_num > 1' or 'in_shape_size ==1' "
            "or 'axes has 0' on nvidia_tensorrt.")

        def teller2(program_config, predictor_config):
            if self.get_nnadapter_device_name() == "intel_openvino":
                input_num = len(program_config.ops[0].inputs)
                if input_num != 1:
                    return True

        self.add_ignore_check_case(
            teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Intel OpenVINO does not support 'input_num > 1'.")

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=150)


if __name__ == "__main__":
    unittest.main(argv=[''])
