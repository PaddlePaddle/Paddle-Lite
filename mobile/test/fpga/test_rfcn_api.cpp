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
#include <fstream>
#include <iomanip>
#include <iostream>
#include "../../src/io/paddle_inference_api.h"

using namespace paddle_mobile;        // NOLINT
using namespace paddle_mobile::fpga;  // NOLINT

static const char *g_image = "../models/rfcn/data.bin";
static const char *g_model = "../models/rfcn/model";
static const char *g_param = "../models/rfcn/params";

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

PaddleMobileConfig GetConfig1() {
  PaddleMobileConfig config;
  config.precision = PaddleMobileConfig::FP32;
  config.device = PaddleMobileConfig::kFPGA;
  config.model_dir = "../models/resnet50";
  config.thread_num = 1;
  config.batch_size = 1;
  config.optimize = true;
  config.quantification = false;
  return config;
}

int main() {
  open_device();
#if 0
  PaddleMobileConfig config1 = GetConfig1();
  auto predictor1 =
      CreatePaddlePredictor<PaddleMobileConfig,
                            PaddleEngineKind::kPaddleMobile>(config1);

  std::cout << "Finishing loading model" << std::endl;

  int img_length1 = 224 * 224 * 3;
  auto img1 =
      reinterpret_cast<float *>(fpga_malloc(img_length1 * sizeof(float)));

  std::cout << "Finishing initializing data" << std::endl;

  struct PaddleTensor t_img1;

  t_img1.dtypeid = type_id<float>().hash_code();
  t_img1.layout = LAYOUT_HWC;
  t_img1.shape = std::vector<int>({1, 224, 224, 3});
  t_img1.name = "Image information";
  t_img1.data.Reset(img1, img_length1 * sizeof(float));
  predictor1->FeedPaddleTensors({t_img1});
  predictor1->Predict_From_To(0, -1);
  std::cout << "Finishing predicting " << std::endl;

  std::vector<PaddleTensor> v1;         // No need to initialize v
  predictor1->FetchPaddleTensors(&v1);  // Old data in v will be cleared
  std::cout << "Output number is " << v1.size() << std::endl;
  std::cout << "out[0] length " << v1[0].data.length() << std::endl;
  fpga_free(img1);
#endif
  ////////////////////////////

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
  t_img.dtypeid = PaddlekTypeId_t::paddle_float;
  t_img_info.layout = LAYOUT_HWC;
  t_img_info.shape = std::vector<int>({1, 3});
  t_img_info.name = "Image information";
  t_img_info.data.Reset(img_info, 3 * sizeof(float));

  t_img.dtypeid = PaddlekTypeId_t::paddle_float;
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
  std::cout << "Output number is " << v.size() << std::endl;
  std::cout << "out[0] length " << v[0].data.length() << std::endl;
  std::cout << "out[1] length " << v[1].data.length() << std::endl;
  std::cout << "out[2] length " << v[2].data.length() << std::endl;

  auto post_nms = v[0].data.length() / sizeof(float) / 8;
  for (int num = 0; num < post_nms; num++) {
    for (int i = 0; i < 8; i++) {
      auto p = reinterpret_cast<float *>(v[0].data.data());
      std::cout << p[num * 8 + i] << std::endl;
    }
  }
  for (int num = 0; num < post_nms; num++) {
    for (int i = 0; i < 8; i++) {
      auto p = reinterpret_cast<float *>(v[1].data.data());
      std::cout << p[num * 8 + i] << std::endl;
    }
  }
  for (int num = 0; num < post_nms; num++) {
    for (int i = 0; i < 4; i++) {
      auto p = reinterpret_cast<float *>(v[2].data.data());
      std::cout << p[num * 4 + i] << std::endl;
    }
  }
  std::cout << "Finish getting vector values" << std::endl;
  fpga_free(img);

  auto version = fpga::paddle_mobile_version();

  std::cout << "0X0" << std::hex << version << std::endl;

  return 0;
}
