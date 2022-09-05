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


class TestDepthwiseConv2dOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])
        self.enable_testing_on_place(
            TargetType.ARM,
            PrecisionType.FP16,
            DataLayoutType.NCHW,
            thread=[1, 4])
        arm_places = [
            Place(TargetType.ARM, PrecisionType.INT8, DataLayoutType.NCHW),
            Place(TargetType.ARM, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=arm_places, thread=[1, 4])
        x86_valid_places = [
            Place(TargetType.X86, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.X86, PrecisionType.INT8, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=x86_valid_places, thread=[1, 4])
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
            "kunlunxin_xtcl", "nvidia_tensorrt", "intel_openvino"
        ])
        xpu_places = [
            Place(TargetType.XPU, PrecisionType.FP32, DataLayoutType.NCHW)
        ]
        self.enable_testing_on_place(places=xpu_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        input_n = draw(st.integers(min_value=1, max_value=4))
        input_c = draw(st.integers(min_value=1, max_value=128))
        input_h = draw(st.integers(min_value=1, max_value=128))
        input_w = draw(st.integers(min_value=1, max_value=128))
        filter_m = input_c
        filter_c = 1
        filter_h = draw(st.sampled_from([3, 5]))
        filter_w = filter_h
        scale_in = draw(st.floats(min_value=0.001, max_value=0.1))
        scale_out = draw(st.floats(min_value=0.001, max_value=0.1))
        assume(input_h >= filter_h)
        assume(input_w >= filter_w)
        groups = input_c
        paddings = draw(st.sampled_from([[0, 0], [1, 1], [2, 2]]))
        dilations = draw(st.sampled_from([[1, 1]]))
        padding_algorithm = draw(st.sampled_from(["VALID", "SAME"]))
        strides = draw(
            st.lists(
                st.integers(
                    min_value=1, max_value=10), min_size=2, max_size=2))
        data_format = "NCHW"
        use_mkldnn = False
        if self.get_target() == "X86":
            use_mkldnn = True

        def calc_output_size():
            if padding_algorithm == "SAME":
                output_h = input_h
                output_w = input_w
            elif padding_algorithm == "VALID":
                output_h = (input_h - (dilations[0] *
                                       (filter_h - 1) + 1)) / strides[0] + 1
                output_w = (input_w - (dilations[1] *
                                       (filter_w - 1) + 1)) / strides[1] + 1
            else:
                output_h = (input_h + 2 * paddings[0] -
                            (dilations[0] *
                             (filter_h - 1) + 1)) / strides[0] + 1
                output_w = (input_w + 2 * paddings[1] -
                            (dilations[1] *
                             (filter_w - 1) + 1)) / strides[1] + 1
            return output_h, output_w

        output_h, output_w = calc_output_size()
        assume(output_h >= 1)
        assume(output_w >= 1)

        in_shape = [input_n, input_c, input_h, input_w]
        weight_shape = [filter_m, filter_c, filter_h, filter_w]

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_filter(*args, **kwargs):
            return np.random.random(weight_shape).astype(np.float32)

        def generate_bias(*args, **kwargs):
            return np.random.random([filter_m]).astype(np.float32)

        def generate_zero_bias(*args, **kwargs):
            return np.zeros([filter_m]).astype(np.float32)

        inputs_type = {"Input": ["input_data"], "Filter": ["filter_data"]}
        inputs_data = {
            "input_data": TensorConfig(data_gen=partial(generate_input))
        }
        weights_data = {
            "filter_data": TensorConfig(data_gen=partial(generate_filter))
        }
        if use_mkldnn:
            has_bias = draw(st.booleans())
            if has_bias:
                inputs_type["Bias"] = ["bias_data"]
                weights_data['bias_data'] = TensorConfig(
                    data_gen=partial(generate_bias))

        if self.get_target() == "XPU":
            inputs_data["bias_data"] = TensorConfig(
                data_gen=partial(generate_zero_bias))
            inputs_type["Bias"] = ["bias_data"]

        depthwise_conv2d_op = OpConfig(
            type="depthwise_conv2d",
            inputs=inputs_type,
            outputs={"Output": ["output_data"]},
            attrs={
                "strides": strides,
                "paddings": paddings,
                "use_mkldnn": use_mkldnn,
                "padding_algorithm": padding_algorithm,
                "groups": groups,
                "dilations": dilations,
                "Scale_in": scale_in,
                "Scale_out": scale_out,
                "data_format": data_format,
            })
        program_config = ProgramConfig(
            ops=[depthwise_conv2d_op],
            weights=weights_data,
            inputs=inputs_data,
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        if target_str == "XPU":
            atol, rtol = 1e-3, 1e-3
        elif self.get_nnadapter_device_name() == "kunlunxin_xtcl":
            atol, rtol = 1e-3, 1e-3
        return self.get_predictor_configs(), ["depthwise_conv2d"], (atol, rtol)

    def add_ignore_pass_case(self):
        def skip_bias_teller(program_config, predictor_config):
            if self.get_target() == "XPU":
                return False
            if "Bias" in program_config.ops[0].inputs.keys():
                return True
            return False

        def _teller1(program_config, predictor_config):
            nnadapter_device_name = self.get_nnadapter_device_name()
            strides = program_config.ops[0].attrs["strides"]
            if nnadapter_device_name == "nvidia_tensorrt":
                return True

        self.add_ignore_check_case(
            skip_bias_teller, IgnoreReasons.PADDLE_NOT_SUPPORT,
            "When paddle is opening the use_mkldnn flag, the kernel implementation of depthwise_conv2d is not registered, so depthwise_conv2d will execute on cpu, the kernel of cpu doesn't support bias, need paddle fix!"
        )

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "The paddle's and trt_layer's results has diff in a specific case on TensorRT. We need to fix it as soon as possible."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
