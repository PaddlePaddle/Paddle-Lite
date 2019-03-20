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

#ifndef PADDLE_MOBILE_FPGA
#define PADDLE_MOBILE_FPGA
#endif

#include "../test_helper.h"
#include "../test_include.h"
#ifdef PADDLE_MOBILE_FPGA_V1
#include "fpga/V1/api.h"
#endif
#ifdef PADDLE_MOBILE_FPGA_V2
#include "fpga/V2/api.h"
#endif

#include <fstream>
#include <iostream>
#include "../../src/io/paddle_inference_api.h"

using namespace paddle_mobile;        // NOLINT
using namespace paddle_mobile::fpga;  // NOLINT

static const char *g_image = "../models/marker/marker1/image.bin";
static const char *g_model = "../models/marker/marker1/model";
static const char *g_param = "../models/marker/marker1/params";

void readStream(std::string filename, char *buf) {
  std::ifstream in;
  in.open(filename, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    std::cout << "open File Failed." << std::endl;
    return;
  }

  in.seekg(0, std::ios::end);  // go to the end
  auto length = in.tellg();    // report location (this is the length)
  in.seekg(0, std::ios::beg);  // go back to the beginning
  in.read(buf, length);
  in.close();
}

PaddleMobileConfig GetConfig() {
  PaddleMobileConfig config;
  config.precision = PaddleMobileConfig::FP32;
  config.device = PaddleMobileConfig::kFPGA;
  config.prog_file = g_model;
  config.param_file = g_param;
  config.thread_num = 1;
  config.batch_size = 1;
  config.optimize = true;
  config.lod_mode = true;
  config.quantification = false;
  return config;
}

int main() {
  open_device();

  PaddleMobileConfig config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<PaddleMobileConfig,
                            PaddleEngineKind::kPaddleMobile>(config);

  std::cout << "Finishing loading model" << std::endl;

  float img_info[3] = {432, 1280, 1.0f};
  int img_length = 432 * 1280 * 3;
  auto img = reinterpret_cast<float *>(fpga_malloc(img_length * sizeof(float)));
  readStream(g_image, reinterpret_cast<char *>(img));

  std::cout << "Finishing initializing data" << std::endl;
  struct PaddleTensor t_img_info, t_img;
  t_img.dtypeid = typeid(float);
  t_img_info.layout = LAYOUT_HWC;
  t_img_info.shape = std::vector<int>({1, 3});
  t_img_info.name = "Image information";
  t_img_info.data.Reset(img_info, 3 * sizeof(float));

  t_img.dtypeid = typeid(float);
  t_img.layout = LAYOUT_HWC;
  t_img.shape = std::vector<int>({1, 432, 1280, 3});
  t_img.name = "Image information";
  t_img.data.Reset(img, img_length * sizeof(float));
  predictor->FeedPaddleTensors({t_img_info, t_img});

  std::cout << "Finishing feeding data " << std::endl;

  predictor->Predict_From_To(0, -1);
  std::cout << "Finishing predicting " << std::endl;

  std::vector<PaddleTensor> v;        // No need to initialize v
  predictor->FetchPaddleTensors(&v);  // Old data in v will be cleared
  for (int i = 0; i < v.size(); ++i) {
    auto p = reinterpret_cast<float *>(v[i].data.data());
    int len = v[i].data.length();
    float result = 0.0f;
    std::string str = "fetch" + std::to_string(i);
    fpga::savefile<float>(str, p, len, result);
  }

  std::cout << "Finish getting vector values" << std::endl;

  ////////////////////////////////////////////////////

  // PaddleTensor tensor;
  // predictor->GetPaddleTensor("fetch2", &tensor);
  // for (int i = 0; i < post_nms; i++) {
  // auto p = reinterpret_cast<float *>(tensor.data.data());
  // std::cout << p[+i] << std::endl;
  // }

  return 0;
}
