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


def update_padding_and_dilation(input_h, input_w, filter_h, filter_w, paddings,
                                padding_algorithm, dilations, strides):
    update_paddings = [paddings[0], paddings[0], paddings[1], paddings[1]]
    update_dilations = dilations
    if padding_algorithm == 'SAME':
        output_h = (input_h + strides[0] - 1) // strides[0]
        output_w = (input_w + strides[1] - 1) // strides[1]
        pad_h_sum = (output_h - 1) * strides[0] + filter_h - input_h
        if pad_h_sum < 0:
            pad_h_sum = 0
        pad_h_0 = pad_h_sum // 2
        pad_h_1 = pad_h_sum - pad_h_0
        pad_w_sum = (output_w - 1) * strides[1] + filter_w - input_w
        if pad_w_sum < 0:
            pad_w_sum = 0
        pad_w_0 = pad_w_sum // 2
        pad_w_1 = pad_w_sum - pad_w_0
        update_paddings = [pad_h_0, pad_h_1, pad_w_0, pad_w_1]
        update_dilations = [1, 1]
    elif padding_algorithm == 'VALID':
        update_paddings = [0, 0, 0, 0]
    return update_paddings, update_dilations


class TestConv2dTransposeOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        x86_valid_places = [
            Place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=x86_valid_places, thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        # self.enable_testing_on_place(
        #     TargetType.ARM,
        #     PrecisionType.FP16,
        #     DataLayoutType.NCHW,
        #     thread=[1, 4])
        arm_valid_places = [
            Place(TargetType.ARM, PrecisionType.INT8, DataLayoutType.NCHW),
            Place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=arm_valid_places, thread=[1, 4])

        opencl_valid_places = [
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
        self.enable_testing_on_place(places=opencl_valid_places)
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=[
            "cambricon_mlu", "nvidia_tensorrt", "intel_openvino"
        ])

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        input_n = draw(st.integers(min_value=1, max_value=4))
        input_c = draw(st.integers(min_value=1, max_value=64))
        input_h = draw(st.integers(min_value=1, max_value=64))
        input_w = draw(st.integers(min_value=1, max_value=64))
        filter_m = draw(st.integers(min_value=1, max_value=64))
        filter_c = input_c
        filter_h = draw(st.integers(min_value=1, max_value=7))
        filter_w = draw(st.integers(min_value=1, max_value=7))
        assume(input_h >= filter_h)
        assume(input_w >= filter_w)
        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=16), min_size=2, max_size=2))
        paddings = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=16), min_size=2, max_size=2))
        output_padding = []
        output_size = []
        groups = draw(st.integers(min_value=1, max_value=input_c))
        assume(filter_c % groups == 0)
        assume(filter_m >= groups and filter_m % groups == 0)
        assume(groups != filter_m or groups != filter_c)
        data_format = draw(st.sampled_from(['NCHW']))
        padding_algorithm = draw(st.sampled_from(['VALID', 'SAME']))
        dilations = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=16), min_size=2, max_size=2))
        use_mkldnn = False
        if self.get_target() == "X86":
            err_info = "Paddle will crash if use_mkldnn == True, need paddle fix!"
            # use_mkldnn = True

        has_output_padding = draw(st.booleans())
        has_output_size = draw(st.booleans())
        paddings, dilations = update_padding_and_dilation(
            input_h=input_h,
            input_w=input_w,
            filter_h=filter_h,
            filter_w=filter_w,
            paddings=paddings,
            padding_algorithm=padding_algorithm,
            dilations=dilations,
            strides=strides)

        def check_constrains(*args, **kwargs):
            '''
                Note: This function will check all constraints that cause conv2d_transpose execution errors, 
                    The following requirements need to be met when generating random data
                    1) output_h, output_w should greater than one.
                    2) if has output_size attr:
                       output_size of Op(ConvTransposeOp) should not be less than the infered output size,
                       output_size of Op(ConvTransposeOp) should be less than infered size + stride
                    3) if has output_padding attr:
                       output_padding[i] < max(strides[i], dilations[i])
                    4) im2col limit:
                        input_h == (output_h + output_padding_h + pad_h_top + pad_h_bottom - (dilation_h * (filter_h - 1) + 1)) / stride_h + 1
                        input_w == (output_w + output_padding_w + pad_w_left + pad_h_right - (dilation_w * (filter_w - 1) + 1)) / stride_w + 1
            '''

            def calc_output_size():
                filter_h_extent = dilations[0] * (filter_h - 1) + 1
                filter_w_extent = dilations[1] * (filter_w - 1) + 1
                infer_output_h = (
                    input_h - 1
                ) * strides[0] - paddings[0] - paddings[1] + filter_h_extent
                infer_output_w = (
                    input_w - 1
                ) * strides[1] - paddings[2] - paddings[3] + filter_w_extent
                return infer_output_h, infer_output_w

            infer_output_h, infer_output_w = calc_output_size()
            assume(infer_output_h >= 1)
            assume(infer_output_w >= 1)

            if has_output_size:
                output_h = draw(
                    st.integers(
                        min_value=infer_output_h,
                        max_value=infer_output_h + strides[0] - 1))
                output_w = draw(
                    st.integers(
                        min_value=infer_output_w,
                        max_value=infer_output_w + strides[1] - 1))
                nonlocal output_size
                output_size = [output_h, output_w]

            if has_output_padding:
                output_padding_h = draw(
                    st.integers(
                        min_value=0,
                        max_value=max(strides[0], dilations[0]) - 1))
                output_padding_w = draw(
                    st.integers(
                        min_value=0,
                        max_value=max(strides[1], dilations[1]) - 1))
                conv_output_h = (infer_output_h + output_padding_h +
                                 paddings[0] + paddings[1] -
                                 (dilations[0] *
                                  (filter_h - 1) + 1)) / strides[0] + 1
                assume(int(conv_output_h) == input_h)
                conv_output_w = (infer_output_w + output_padding_w +
                                 paddings[2] + paddings[3] -
                                 (dilations[1] *
                                  (filter_w - 1) + 1)) / strides[1] + 1
                assume(int(conv_output_w) == input_w)
                nonlocal output_padding
                output_padding = [output_padding_h, output_padding_w]

        check_constrains()

        input_shape = []
        if data_format == 'NCHW':
            input_shape = [input_n, input_c, input_h, input_w]
        elif data_format == 'NHWC':
            input_shape = [input_n, input_h, input_w, input_c]
        weight_shape = [filter_c, filter_m // groups, filter_h,
                        filter_w]  # data_format = 'CMHW'

        def generate_input(*args, **kwargs):
            return np.random.random(input_shape).astype(np.float32)

        def generate_filter(*args, **kwargs):
            return np.random.random(weight_shape).astype(np.float32)

        def generate_bias(*args, **kwargs):
            return np.random.random([filter_m]).astype(np.float32)

        inputs_data = {
            "input_data": TensorConfig(data_gen=partial(generate_input))
        }
        inputs_type = {"Input": ["input_data"], "Filter": ["filter_data"]}
        if use_mkldnn:
            inputs_data["bias_data"] = TensorConfig(
                data_gen=partial(generate_bias))
            inputs_type["Bias"] = ["bias_data"]

        conv2d_transpose_op = OpConfig(
            type="conv2d_transpose",
            inputs=inputs_type,
            outputs={"Output": ["output_data"]},
            attrs={
                "output_padding": output_padding,
                "output_size": output_size,
                "groups": groups,
                "dilations": dilations,
                "strides": strides,
                "paddings": paddings,
                "data_format": data_format,
                "padding_algorithm": padding_algorithm,
                "use_mkldnn": use_mkldnn,
                "is_test": True
            })

        program_config = ProgramConfig(
            ops=[conv2d_transpose_op],
            weights={
                "filter_data": TensorConfig(data_gen=partial(generate_filter)),
            },
            inputs=inputs_data,
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        configs = self.get_predictor_configs()
        if target_str == "OpenCL":
            atol, rtol = 1e-4, 1e-4
        if self.get_nnadapter_device_name() == "nvidia_tensorrt":
            for config in configs:
                if config.target() == TargetType.NNAdapter:
                    config.set_nnadapter_context_properties(
                        "NVIDIA_TENSORRT_MAX_WORKSPACE_SIZE=1073741824")
        return configs, ["conv2d_transpose"], (atol, rtol)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            groups = program_config.ops[0].attrs["groups"]

            if predictor_config.target(
            ) == TargetType.ARM and predictor_config.precision(
            ) == PrecisionType.INT8 and groups > 1:
                return True

        def _teller2(program_config, predictor_config):
            groups = program_config.ops[0].attrs["groups"]
            output_size = program_config.ops[0].attrs["output_size"]
            output_padding = program_config.ops[0].attrs["output_padding"]
            filter_shape = list(program_config.weights["filter_data"].shape)
            if self.get_nnadapter_device_name() == "nvidia_tensorrt":
                if (filter_shape[1] % groups != 0
                    ) or len(output_size) != 0 or len(
                        output_padding) != 0 or filter_shape[
                            2] != filter_shape[3]:
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.ACCURACY_ERROR,
            "Lite has diff in a specific case on arm-int8. We need to fix it as soon as possible."
        )

        self.add_ignore_check_case(
            _teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support output_size or output_padding or filter_shape[2] != filter_shape[3] on nvidia_tensorrt, and group count must divide output channel count."
        )

        def _teller3(program_config, predictor_config):
            groups = program_config.ops[0].attrs["groups"]
            filter_shape = list(program_config.weights["filter_data"].shape)
            if "intel_openvino" in self.get_nnadapter_device_name():
                if (filter_shape[1] % groups) != 0:
                    return True

        self.add_ignore_check_case(
            _teller3, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Group count must divide output channel count in intel OpenVINO.")

        def _teller4(program_config, predictor_config):
            target_type = predictor_config.target()
            if target_type == TargetType.OpenCL:
                return True

        self.add_ignore_check_case(
            _teller4, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "conv_transpose_image_compute.cc:144 PrepareForRun] \
                                   conv2d_transpose image compute not support this condition yet! 1 2 11"
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=150)


if __name__ == "__main__":
    unittest.main(argv=[''])
