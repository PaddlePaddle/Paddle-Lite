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

  config.prog_file = "../models/imagefixmodel/model";
  config.param_file = "../models/imagefixmodel/params";
  config.lod_mode = false;
  config.load_when_predict = false;
  return config;
}

int main() {
  PaddleMobileConfig config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<PaddleMobileConfig,
                            PaddleEngineKind::kPaddleMobile>(config);

  // factor
  int input_rgb_len = 1 * 3 * 256 * 256;
  std::vector<float> input_rgb_v(input_rgb_len, 1);
  // SetupData<float>(input_rgb_v.data(), input_rgb_len, 0.f, 1.f);

  PaddleTensor input_rgb;
  input_rgb.shape = std::vector<int>({1, 3, 256, 256});
  input_rgb.data = PaddleBuf(input_rgb_v.data(), input_rgb_len * sizeof(float));
  input_rgb.dtype = PaddleDType::FLOAT32;
  input_rgb.layout = LayoutType::LAYOUT_CHW;

  // remap
  int input_mask_len = 1 * 3 * 256 * 256;
  std::vector<float> input_mask_v(input_mask_len, 1);
  // SetupData<float>(input_mask_v.data(), input_mask_len, 0.f, 1.f);

  PaddleTensor input_mask;
  input_mask.shape = std::vector<int>({1, 3, 256, 256});
  input_mask.data =
      PaddleBuf(input_mask_v.data(), input_mask_len * sizeof(float));
  input_mask.dtype = PaddleDType::FLOAT32;
  input_mask.layout = LayoutType::LAYOUT_CHW;

  PaddleTensor output0;
  output0.shape = std::vector<int>({});
  output0.data = PaddleBuf();
  output0.dtype = PaddleDType::FLOAT32;
  output0.layout = LayoutType::LAYOUT_CHW;

  // PaddleTensor output1;
  // output1.shape = std::vector<int>({});
  // output1.data = PaddleBuf();
  // output1.dtype = PaddleDType::FLOAT32;
  // output1.layout = LayoutType::LAYOUT_CHW;

  // PaddleTensor output2;
  // output2.shape = std::vector<int>({});
  // output2.data = PaddleBuf();
  // output2.dtype = PaddleDType::FLOAT32;
  // output2.layout = LayoutType::LAYOUT_CHW;

  // PaddleTensor output3;
  // output3.shape = std::vector<int>({});
  // output3.data = PaddleBuf();
  // output3.dtype = PaddleDType::FLOAT32;
  // output3.layout = LayoutType::LAYOUT_CHW;
  std::cout << "feed : " << std::endl;

  predictor->Feed("input_rgb", input_rgb);

  std::cout << "feed : " << std::endl;

  predictor->Feed("input_mask", input_mask);

  std::cout << "run : " << std::endl;

  predictor->Run();

  std::cout << "fetch : " << std::endl;

  predictor->Fetch("save_infer_model/scale_0", &output0);

  float* out_ptr0 = reinterpret_cast<float*>(output0.data.data());
  std::cout << " print output0 : " << std::endl;
  int numel = output0.data.length() / sizeof(float);
  int stride = numel / 20;
  stride = stride > 0 ? stride : 1;
  for (size_t j = 0; j < numel; j += stride) {
    std::cout << out_ptr0[j] << " ";
  }
  std::cout << std::endl;

  return 0;
}
