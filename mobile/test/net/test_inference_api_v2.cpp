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
  config.pre_post_type = PaddleMobileConfig::NONE_PRE_POST;

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

  std::vector<float> inputv;
  std::vector<int64_t> dims{1, 1, 300, 300};
  GetInput<float>(g_test_image_1x3x224x224, &inputv, dims);

  PaddleTensor input;
  input.shape = std::vector<int>({1, 1, 300, 300});
  input.data = PaddleBuf(inputv.data(), input_length * sizeof(float));
  input.dtype = PaddleDType::FLOAT32;
  input.layout = LayoutType::LAYOUT_CHW;

  PaddleTensor output;
  output.shape = std::vector<int>({});
  output.data = PaddleBuf();
  output.dtype = PaddleDType::FLOAT32;
  output.layout = LayoutType::LAYOUT_CHW;

  float* in_data = inputv.data();
  std::cout << " print input : " << std::endl;
  int stride = input_length / 20;
  stride = stride > 0 ? stride : 1;
  for (size_t j = 0; j < input_length; j += stride) {
    std::cout << in_data[j] << " ";
  }
  std::cout << std::endl;

  predictor->Feed("input_rgb", input);
  predictor->Run();
  predictor->Fetch("save_infer_model/scale_0", &output);

  float* out_data = reinterpret_cast<float*>(output.data.data());
  std::cout << " print output : " << std::endl;
  int numel = output.data.length() / sizeof(float);
  stride = numel / 20;
  stride = stride > 0 ? stride : 1;
  for (size_t j = 0; j < numel; j += stride) {
    std::cout << out_data[j] << " ";
  }
  std::cout << std::endl;

  return 0;
}
