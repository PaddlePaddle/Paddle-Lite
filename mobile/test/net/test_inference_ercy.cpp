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

  config.prog_file = "../models/ercy/model";
  config.param_file = "../models/ercy/params";
  config.lod_mode = false;
  config.load_when_predict = false;
  return config;
}

int main() {
  PaddleMobileConfig config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<PaddleMobileConfig,
                            PaddleEngineKind::kPaddleMobile>(config);

  // reliable
  int re_len = 1 * 1 * 64 * 72;
  std::vector<float> re_v;
  std::vector<int64_t> re_dims{1, 1, 64, 72};
  GetInput<float>(g_test_image_1x3x224x224, &re_v, re_dims);

  PaddleTensor re;
  re.shape = std::vector<int>({1, 1, 64, 72});
  re.data = PaddleBuf(re_v.data(), re_len * sizeof(float));
  re.dtype = PaddleDType::FLOAT32;
  re.layout = LayoutType::LAYOUT_CHW;

  // grid
  int grid_len = 1 * 64 * 72 * 2;
  std::vector<float> grid_v;
  std::vector<int64_t> grid_dims{1, 64, 72, 2};
  GetInput<float>(g_test_image_1x3x224x224, &grid_v, grid_dims);

  PaddleTensor grid;
  grid.shape = std::vector<int>({1, 64, 72, 2});
  grid.data = PaddleBuf(grid_v.data(), grid_len * sizeof(float));
  grid.dtype = PaddleDType::FLOAT32;
  grid.layout = LayoutType::LAYOUT_CHW;

  // last_input
  int last_len = 1 * 128 * 64 * 72;
  std::vector<float> last_v;
  std::vector<int64_t> last_dims{1, 128, 64, 72};
  GetInput<float>(g_test_image_1x3x224x224, &last_v, last_dims);

  PaddleTensor last;
  last.shape = std::vector<int>({1, 128, 64, 72});
  last.data = PaddleBuf(last_v.data(), last_len * sizeof(float));
  last.dtype = PaddleDType::FLOAT32;
  last.layout = LayoutType::LAYOUT_CHW;

  // input_rgb
  int input_rgb_len = 1 * 4 * 256 * 288;
  std::vector<float> input_rgb_v;
  std::vector<int64_t> input_rgb_dims{1, 4, 256, 288};
  GetInput<float>(g_test_image_1x3x224x224, &input_rgb_v, input_rgb_dims);

  PaddleTensor input_rgb;
  input_rgb.shape = std::vector<int>({1, 4, 256, 288});
  input_rgb.data = PaddleBuf(input_rgb_v.data(), input_rgb_len * sizeof(float));
  input_rgb.dtype = PaddleDType::FLOAT32;
  input_rgb.layout = LayoutType::LAYOUT_CHW;

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

  predictor->Feed("reliable", re);
  predictor->Feed("grid", grid);
  predictor->Feed("last_input", last);
  predictor->Feed("input_rgb", input_rgb);
  predictor->Run();
  predictor->Fetch("save_infer_model/scale_0", &output0);
  predictor->Fetch("save_infer_model/scale_1", &output1);

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
