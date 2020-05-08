/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <iostream>
#include "../test_helper.h"
#include "io/paddle_inference_api.h"

using namespace paddle_mobile;  // NOLINT

PaddleMobileConfig GetConfig() {
  PaddleMobileConfig config;
  config.precision = PaddleMobileConfig::FP32;
  config.device = PaddleMobileConfig::kGPU_CL;
  config.pre_post_type = PaddleMobileConfig::UINT8_255;

  config.prog_file = "../models/superv2/model";
  config.param_file = "../models/superv2/params";
  config.lod_mode = false;
  config.load_when_predict = true;
  config.cl_path = "/data/local/tmp/bin";
  return config;
}

int main() {
  PaddleMobileConfig config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<PaddleMobileConfig,
                            PaddleEngineKind::kPaddleMobile>(config);

  int input_length = 1 * 1 * 300 * 300;
  int output_length = input_length;

  uint8_t data_ui[300 * 300];
  for (int i = 0; i < input_length; ++i) {
    data_ui[i] = i % 256;
  }

  PaddleTensor input;
  input.shape = std::vector<int>({1, 1, 300, 300});
  input.data = PaddleBuf(data_ui, sizeof(data_ui));
  input.dtype = PaddleDType::UINT8;
  input.layout = LayoutType::LAYOUT_CHW;
  std::vector<PaddleTensor> inputs(1, input);

  PaddleTensor output;
  output.shape = std::vector<int>({});
  output.data = PaddleBuf();
  output.dtype = PaddleDType::UINT8;
  output.layout = LayoutType::LAYOUT_CHW;
  std::vector<PaddleTensor> outputs(1, output);

  std::cout << " print input : " << std::endl;
  int stride = input_length / 20;
  stride = stride > 0 ? stride : 1;
  for (size_t j = 0; j < input_length; j += stride) {
    std::cout << (unsigned)data_ui[j] << " ";
  }
  std::cout << std::endl;

  predictor->Run(inputs, &outputs);

  std::cout << " print output : " << std::endl;
  uint8_t *data_o = static_cast<uint8_t *>(outputs[0].data.data());
  int numel = outputs[0].data.length() / sizeof(uint8_t);
  stride = numel / 20;
  stride = stride > 0 ? stride : 1;
  for (size_t j = 0; j < numel; j += stride) {
    std::cout << (unsigned)data_o[j] << " ";
  }
  std::cout << std::endl;

  return 0;
}
