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
#include <iostream>
#include "../../src/io/paddle_inference_api.h"

using namespace paddle_mobile;
using namespace paddle_mobile::fpga;

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

int main() {
  open_device();
  PaddleMobileConfig config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<PaddleMobileConfig,
                            PaddleEngineKind::kPaddleMobile>(config);

  std::cout << "Finishing loading model" << std::endl;

  float img_info[3] = {768, 1536, 768.0f / 960.0f};
  int img_length = 768 * 1536 * 3;
  auto img = reinterpret_cast<float *>(fpga_malloc(img_length * sizeof(float)));
  readStream(g_image, reinterpret_cast<char *>(img));

  std::cout << "Finishing initializing data" << std::endl;
  /*
    predictor->FeedData({img_info, img});
    predictor->Predict_From_To(0, -1);
    std::cout << " Finishing predicting " << std::endl;
      std::vector<void *> v(3, nullptr);
      predictor->GetResults(&v);
    int post_nms = 300;
    for (int num = 0; num < post_nms; num ++){
      for (int i = 0; i < 8; i ++){
        std:: cout << ((float*)(v[0]))[num * 8 + i] << std::endl;
      }
    }
    for (int num = 0; num < post_nms; num ++){
      for (int i = 0; i < 8; i ++){
        std:: cout << ((float*)(v[1]))[num * 8 + i] << std::endl;
      }
    }
    for (int num = 0; num < post_nms; num ++){
      for (int i = 0; i < 4; i ++){
        std:: cout << ((float*)(v[2]))[num * 4 + i] << std::endl;
      }
    }
  */

  struct PaddleTensor t_img_info, t_img;
  t_img_info.dtype = FLOAT32;
  t_img_info.layout = LAYOUT_HWC;
  t_img_info.shape = std::vector<int>({1, 3});
  t_img_info.name = "Image information";
  t_img_info.data.Reset(img_info, 3 * sizeof(float));

  t_img.dtype = FLOAT32;
  t_img.layout = LAYOUT_HWC;
  t_img.shape = std::vector<int>({1, 768, 1536, 3});
  t_img.name = "Image information";
  t_img.data.Reset(img, img_length * sizeof(float));
  predictor->FeedPaddleTensors({t_img_info, t_img});

  std::cout << "Finishing feeding data " << std::endl;

  predictor->Predict_From_To(0, -1);
  std::cout << "Finishing predicting " << std::endl;

  std::vector<PaddleTensor> v;        // No need to initialize v
  predictor->FetchPaddleTensors(&v);  // Old data in v will be cleared
  std::cout << "Output number is " << v.size() << std::endl;

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

  PaddleTensor tensor;
  predictor->GetPaddleTensor("fetch2", &tensor);
  for (int i = 0; i < post_nms; i++) {
    auto p = reinterpret_cast<float *>(tensor.data.data());
    std::cout << p[+i] << std::endl;
  }

  return 0;
}
