# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
'''
Paddle-Lite full python api demo
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from paddlelite.lite import *

# Command arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_dir", default="", type=str, help="Non-combined Model dir path")
parser.add_argument(
    "--model_file", default="", type=str, help="Model file")
parser.add_argument(
    "--param_file", default="", type=str, help="Combined model param file")

def RunModel(args):
    # 1. Set config information
    config = CxxConfig()
    if args.model_file != '' and args.param_file != '':
        config.set_model_file(args.model_file)
        config.set_param_file(args.param_file)
    else:
        config.set_model_dir(args.model_dir)
    # For arm platform (armlinux), you can set places = [Place(TargetType.ARM, PrecisionType.FP32)]
    places = [Place(TargetType.X86, PrecisionType.FP32)]
    config.set_valid_places(places)

    # 2. Create paddle predictor
    predictor = create_paddle_predictor(config)

    # 3. Set input data
    input_tensor = predictor.get_input(0)
    input_tensor.resize([1, 3, 224, 224])
    input_tensor.set_float_data([1.] * 3 * 224 * 224)

    # 4. Run model
    predictor.run()

    # 5. Get output data
    output_tensor = predictor.get_output(0)
    print(output_tensor.shape())
    print(output_tensor.float_data()[:10])

if __name__ == '__main__':
    args = parser.parse_args()
    RunModel(args)
