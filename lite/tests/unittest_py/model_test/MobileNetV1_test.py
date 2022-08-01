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
import tempfile
import os
sys.path.append('../')

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
            thread=[1])

    def is_model_test(self) -> bool:
        return True

    def get_model(self):
        # Download and unzip model
        URL = "https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz"
        model_name = "MobileNetV1"
        file_name = "MobileNetV1.tar.gz"

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
            return model, params

    def prepare_input_data(self, draw):
        ih = draw(st.integers(min_value=32, max_value=512))
        oh = ih
        in_shape = [1, 3, ih, oh]
        inputs = {"inputs": TensorConfig(shape=in_shape)}
        return inputs

    def sample_predictor_configs(self):
        return self.get_predictor_configs(), [""], (1e-5, 1e-5)

    def add_ignore_pass_case(self):
        pass

    def test(self, *args, **kwargs):
        model, params = self.get_model()
        self.run_and_statis(
            quant=False,
            model=model,
            params=params,
            min_success_num=1,
            max_examples=10)


if __name__ == "__main__":
    unittest.main(argv=[''])
