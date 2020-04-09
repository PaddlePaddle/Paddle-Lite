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

  config.prog_file = "../models/m2fm/model";
  config.param_file = "../models/m2fm/params";
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
  int factor_len = 1 * 256 * 1 * 1;
  std::vector<float> factor_v;
  std::vector<int64_t> factor_dims{1, 256, 1, 1};
  GetInput<float>(g_test_image_1x3x224x224, &factor_v, factor_dims);

  PaddleTensor factor;
  factor.shape = std::vector<int>({1, 256, 1, 1});
  factor.data = PaddleBuf(factor_v.data(), factor_len * sizeof(float));
  factor.dtype = PaddleDType::FLOAT32;
  factor.layout = LayoutType::LAYOUT_CHW;

  // remap
  int remap_len = 1 * 256 * 256 * 2;
  std::vector<float> remap_v;
  std::vector<int64_t> remap_dims{1, 256, 256, 2};
  GetInput<float>(g_test_image_1x3x224x224, &remap_v, remap_dims);

  PaddleTensor remap;
  remap.shape = std::vector<int>({1, 256, 256, 2});
  remap.data = PaddleBuf(remap_v.data(), remap_len * sizeof(float));
  remap.dtype = PaddleDType::FLOAT32;
  remap.layout = LayoutType::LAYOUT_CHW;

  // image
  int image_len = 1 * 3 * 256 * 256;
  std::vector<float> image_v;
  std::vector<int64_t> image_dims{1, 3, 256, 256};
  GetInput<float>(g_test_image_1x3x224x224, &image_v, image_dims);

  PaddleTensor image;
  image.shape = std::vector<int>({1, 3, 256, 256});
  image.data = PaddleBuf(image_v.data(), image_len * sizeof(float));
  image.dtype = PaddleDType::FLOAT32;
  image.layout = LayoutType::LAYOUT_CHW;

  PaddleTensor output0;
  output0.shape = std::vector<int>({});
  output0.data = PaddleBuf();
  output0.dtype = PaddleDType::FLOAT32;
  output0.layout = LayoutType::LAYOUT_CHW;

  PaddleTensor output1;
  output1.shape = std::vector<int>({});
  output1.data = PaddleBuf();
  output1.dtype = PaddleDType::FLOAT32;
  output1.layout = LayoutType::LAYOUT_CHW;

  PaddleTensor output2;
  output2.shape = std::vector<int>({});
  output2.data = PaddleBuf();
  output2.dtype = PaddleDType::FLOAT32;
  output2.layout = LayoutType::LAYOUT_CHW;

  PaddleTensor output3;
  output3.shape = std::vector<int>({});
  output3.data = PaddleBuf();
  output3.dtype = PaddleDType::FLOAT32;
  output3.layout = LayoutType::LAYOUT_CHW;

  predictor->Feed("x2paddle_mul_factor", factor);
  predictor->Feed("x2paddle_base_remap", remap);
  predictor->Feed("x2paddle_image", image);
  predictor->Run();
  predictor->Fetch("save_infer_model/scale_0", &output0);
  predictor->Fetch("save_infer_model/scale_1", &output1);
  predictor->Fetch("save_infer_model/scale_2", &output2);
  predictor->Fetch("save_infer_model/scale_3", &output3);

  float* out_ptr0 = reinterpret_cast<float*>(output0.data.data());
  float* out_ptr1 = reinterpret_cast<float*>(output1.data.data());
  std::cout << " print output0 : " << std::endl;
  int numel = output0.data.length() / sizeof(float);
  int stride = numel / 20;
  stride = stride > 0 ? stride : 1;
  for (size_t j = 0; j < numel; j += stride) {
    std::cout << out_ptr0[j] << " ";
  }
  std::cout << std::endl;

  std::cout << " print output1 : " << std::endl;
  numel = output1.data.length() / sizeof(float);
  stride = numel / 20;
  stride = stride > 0 ? stride : 1;
  for (size_t j = 0; j < numel; j += stride) {
    std::cout << out_ptr1[j] << " ";
  }
  std::cout << std::endl;

  return 0;
}
