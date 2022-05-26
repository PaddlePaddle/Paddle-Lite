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

import os
import argparse

URL = "url"
MODEL_NAME = "model_name"
FILE_NAME = "file_name"
INPUT_SHAPES = "input_shapes"

all_configs = []

MobileNetV1_config = {
    "url":
    "https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV1.tar.gz",
    "model_name": "MobileNetV1",
    "file_name": "MobileNetV1.tar.gz",
    "input_shapes": ["1,3,224,224"]
}

MobileNetV2_config = {
    "url":
    "https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV2.tar.gz",
    "model_name": "MobileNetV2",
    "file_name": "MobileNetV2.tar.gz",
    "input_shapes": ["1,3,224,224"]
}

MobileNetV3_large_x1_0_config = {
    "url":
    "https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV3_large_x1_0.tar.gz",
    "model_name": "MobileNetV3_large_x1_0",
    "file_name": "MobileNetV3_large_x1_0.tar.gz",
    "input_shapes": ["1,3,224,224"]
}

MobileNetV3_small_x1_0_config = {
    "url":
    "https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/MobileNetV3_small_x1_0.tar.gz",
    "model_name": "MobileNetV3_small_x1_0",
    "file_name": "MobileNetV3_small_x1_0.tar.gz",
    "input_shapes": ["1,3,224,224"]
}

ResNet50_config = {
    "url":
    "https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/ResNet50.tar.gz",
    "model_name": "ResNet50",
    "file_name": "ResNet50.tar.gz",
    "input_shapes": ["1,3,224,224"]
}

ssdlite_mobilenet_v3_large_config = {
    "url":
    "https://paddle-inference-dist.bj.bcebos.com/AI-Rank/mobile/ssdlite_mobilenet_v3_large.tar.gz",
    "model_name": "ssdlite_mobilenet_v3_large",
    "file_name": "ssdlite_mobilenet_v3_large.tar.gz",
    "input_shapes": ["1,3,320,320"]
}

all_configs.append(MobileNetV1_config)
all_configs.append(MobileNetV2_config)
all_configs.append(MobileNetV3_large_x1_0_config)
all_configs.append(MobileNetV3_small_x1_0_config)
all_configs.append(ResNet50_config)
all_configs.append(ssdlite_mobilenet_v3_large_config)

parser = argparse.ArgumentParser()
parser.add_argument("--target", help="set target, default=X86", default="X86")
args = parser.parse_args()

for config in all_configs:
    input_info_str = ""
    for input_shape in config[INPUT_SHAPES]:
        input_info_str = input_info_str + " --input_shapes={}".format(
            input_shape)
        if args.target == "X86":
            command = "python3.7 {}/model_test_base.py --target=X86 --url={} --model_name={} --file_name={} {}".format(
                os.getcwd(), config[URL], config[MODEL_NAME],
                config[FILE_NAME], input_info_str)
        elif args.target == "Host":
            command = "python3.7 {}/model_test_base.py --target=Host --url={} --model_name={} --file_name={} {}".format(
                os.getcwd(), config[URL], config[MODEL_NAME],
                config[FILE_NAME], input_info_str)
        elif args.target == "ARM":
            command = "python3.8 {}/model_test_base.py --target=ARM --url={} --model_name={} --file_name={} {}".format(
                os.getcwd(), config[URL], config[MODEL_NAME],
                config[FILE_NAME], input_info_str)
        elif args.target == "OpenCL":
            command = "python3.8 {}/model_test_base.py --target=OpenCL --url={} --model_name={} --file_name={} {}".format(
                os.getcwd(), config[URL], config[MODEL_NAME],
                config[FILE_NAME], input_info_str)
        elif args.target == "Metal":
            command = "python3.8 {}/model_test_base.py --target=Metal --url={} --model_name={} --file_name={} {}".format(
                os.getcwd(), config[URL], config[MODEL_NAME],
                config[FILE_NAME], input_info_str)
    print(command)
    os.system(command)
