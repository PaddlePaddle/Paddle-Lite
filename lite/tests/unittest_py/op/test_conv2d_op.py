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
import numpy as np
from functools import partial


class TestConv2dOp(AutoScanTest):
    def __init__(self, *args, **kwargs):
        AutoScanTest.__init__(self, *args, **kwargs)
        self.enable_testing_on_place(
            TargetType.X86,
            PrecisionType.FP32,
            DataLayoutType.NCHW,
            thread=[1, 4])

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
        metal_places = [
            Place(TargetType.Metal, PrecisionType.FP16,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.Metal, PrecisionType.FP32,
                  DataLayoutType.MetalTexture2DArray),
            Place(TargetType.ARM, PrecisionType.FP32),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=metal_places)
        self.enable_testing_on_place(TargetType.NNAdapter, PrecisionType.FP32)
        self.enable_devices_on_nnadapter(device_names=[
            "kunlunxin_xtcl", "cambricon_mlu", "nvidia_tensorrt",
            "intel_openvino"
        ])
        xpu_places = [
            Place(TargetType.XPU, PrecisionType.FP32, DataLayoutType.NCHW),
            Place(TargetType.Host, PrecisionType.FP32)
        ]
        self.enable_testing_on_place(places=xpu_places)

    def is_program_valid(self,
                         program_config: ProgramConfig,
                         predictor_config: CxxConfig) -> bool:
        return True

    def sample_program_configs(self, draw):
        num = draw(st.integers(min_value=1, max_value=4))
        cin = draw(st.integers(min_value=1, max_value=128))
        cout = draw(st.integers(min_value=1, max_value=128))
        height = draw(st.integers(min_value=1, max_value=128))
        width = draw(st.integers(min_value=1, max_value=128))
        kw = draw(st.integers(min_value=1, max_value=5))
        kh = draw(st.integers(min_value=1, max_value=5))
        groups = draw(st.integers(min_value=1, max_value=128))
        scale_in = draw(st.floats(min_value=0.001, max_value=0.1))
        scale_out = draw(st.floats(min_value=0.001, max_value=0.1))
        assume(cin % groups == 0)
        assume(cout % groups == 0)
        w_cin = (int)(cin / groups)
        in_shape = [num, cin, height, width]
        weight_shape = [cout, w_cin, kh, kw]
        assume(in_shape[2] >= weight_shape[2])
        assume(in_shape[3] >= weight_shape[3])

        paddings = draw(
            st.lists(
                st.integers(
                    min_value=0, max_value=2), min_size=2, max_size=2))
        dilations = draw(st.sampled_from([[1, 1]]))
        padding_algorithm = draw(st.sampled_from(["VALID", "SAME"]))
        strides = draw(st.sampled_from([[1, 1], [2, 2]]))
        data_format = "NCHW"
        use_mkldnn = False
        if self.target[0] == "X86":
            use_mkldnn = True

        def generate_input(*args, **kwargs):
            return np.random.random(in_shape).astype(np.float32)

        def generate_filter(*args, **kwargs):
            return np.random.random(weight_shape).astype(np.float32)

        def generate_bias(*args, **kwargs):
            return np.random.random([cout]).astype(np.float32)

        def generate_zero_bias(*args, **kwargs):
            return np.zeros([cout]).astype(np.float32)

        inputs_data = {
            "input_data": TensorConfig(data_gen=partial(generate_input))
        }
        inputs_type = {"Input": ["input_data"], "Filter": ["filter_data"]}
        if use_mkldnn:
            inputs_data["bias_data"] = TensorConfig(
                data_gen=partial(generate_bias))
            inputs_type["Bias"] = ["bias_data"]

        if self.get_target() == "XPU":
            inputs_data["bias_data"] = TensorConfig(
                data_gen=partial(generate_zero_bias))
            inputs_type["Bias"] = ["bias_data"]

        conv_op = OpConfig(
            type="conv2d",
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
                "data_format": data_format
            })
        program_config = ProgramConfig(
            ops=[conv_op],
            weights={
                "filter_data": TensorConfig(data_gen=partial(generate_filter))
            },
            inputs=inputs_data,
            outputs=["output_data"])
        return program_config

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        target_str = self.get_target()
        if target_str == "Metal":
            atol, rtol = 1e-3, 1e-3
        elif target_str == "XPU":
            atol, rtol = 1e-3, 1e-3
        elif self.get_nnadapter_device_name() == "kunlunxin_xtcl":
            atol, rtol = 1e-3, 1e-3
        return self.get_predictor_configs(), ["conv2d"], (atol, rtol)

    def add_ignore_pass_case(self):
        def _teller1(program_config, predictor_config):
            target_type = predictor_config.target()
            precision_type = predictor_config.precision()
            input_shape = program_config.inputs["input_data"].shape
            filter_data = program_config.weights["filter_data"].shape
            stride_data = program_config.ops[0].attrs["strides"]
            groups = program_config.ops[0].attrs["groups"]
            if target_type == TargetType.ARM and precision_type == PrecisionType.INT8:
                if input_shape[1] > 80 and filter_data[2] == 3 and filter_data[
                        3] == 3 and groups == 1 and stride_data[
                            0] == 1 and stride_data[1] == 1:
                    return True

        def _teller2(program_config, predictor_config):
            target_type = predictor_config.target()
            input_shape = program_config.inputs["input_data"].shape
            filter_data = program_config.weights["filter_data"].shape
            groups = program_config.ops[0].attrs["groups"]
            if target_type == TargetType.Metal:
                if groups != 1:
                    return True
                if input_shape[0] != 1 or input_shape[1] < 3 or filter_data[
                        0] < 3:
                    return True

        def _teller3(program_config, predictor_config):
            groups = program_config.ops[0].attrs["groups"]
            filter_shape = list(program_config.weights["filter_data"].shape)
            if self.get_nnadapter_device_name() == "nvidia_tensorrt":
                if (groups > 1 and filter_shape[0] != groups) \
                    or filter_shape[2] != filter_shape[3]:
                    return True

        self.add_ignore_check_case(
            _teller1, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Input data is 0-1, int8 winograd will overflow when input channel is more than 80."
        )
        self.add_ignore_check_case(
            _teller2, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support this op in a specific case on metal. We need to fix it as soon as possible."
        )
        self.add_ignore_check_case(
            _teller3, IgnoreReasons.PADDLELITE_NOT_SUPPORT,
            "Lite does not support this op in a specific case on TensorRT. We need to fix it as soon as possible."
        )

    def test(self, *args, **kwargs):
        self.run_and_statis(quant=False, max_examples=300)


if __name__ == "__main__":
    unittest.main(argv=[''])
