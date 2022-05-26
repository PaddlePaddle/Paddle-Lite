# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
import tempfile
import os
sys.path.append('../')

import paddle.inference as paddle_infer
from auto_scan_test import AutoScanTest, IgnoreReasons
from program_config import TensorConfig, ProgramConfig, OpConfig, CxxConfig, TargetType, PrecisionType, DataLayoutType, Place
import unittest

import hypothesis
from hypothesis import given, settings, seed, example, assume
import hypothesis.strategies as st


class TestMobileNetV1(AutoScanTest):
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

    def is_model_test(self) -> bool:
        return True

    def get_model(self):
        # Download and unzip model
        URL = self.args.url
        model_name = self.args.model_name
        file_name = self.args.file_name

        # Save model in temporary directory and load model into memory
        with tempfile.TemporaryDirectory() as tmpdirname:
            os.system("wget -P {} {}".format(tmpdirname, URL))
            os.system("tar -zvxf {0}/{1} -C {0}".format(tmpdirname, file_name))
            with open("{}/{}/inference.pdmodel".format(tmpdirname, model_name),
                      "rb") as f:
                model = f.read()
            with open("{}/{}/inference.pdiparams".format(
                    tmpdirname, model_name), "rb") as f:
                params = f.read()

            # acquire input names
            paddle_config = self.create_inference_config(ir_optim=False)
            paddle_config.set_model_buffer(model,
                                           len(model), params, len(params))
            predictor = paddle_infer.create_predictor(paddle_config)
            self.input_names = predictor.get_input_names()

            return model, params

    def strList_to_intList(self, input_shape):
        out = []
        for i in input_shape.split(","):
            out.append(int(i))
        return out

    def prepare_input_data(self, draw):
        inputs = {}
        assert len(self.input_names) == len(self.args.input_shapes)
        for i in range(len(self.input_names)):
            input_shape = self.strList_to_intList(self.args.input_shapes[i])
            inputs[self.input_names[i]] = TensorConfig(shape=input_shape)
        return inputs

    def sample_predictor_configs(self):
        atol, rtol = 1e-5, 1e-5
        if self.args.model_name == "ssdlite_mobilenet_v3_large":
            atol, rtol = 2e-3, 2e-3
        return self.get_predictor_configs(), [""], (atol, rtol)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        model, params = self.get_model()
        self.run_and_statis(
            quant=False,
            model=model,
            params=params,
            min_success_num=1,
            max_examples=1)


if __name__ == "__main__":
    unittest.main(argv=[''])
