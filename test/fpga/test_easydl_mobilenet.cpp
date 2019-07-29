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
#include <fstream>
#include <iomanip>
#include <iostream>
#include "../test_include.h"

// #ifdef PADDLE_MOBILE_FPGA_KD
// #include "fpga/KD/api.h"
// #endif
#include "fpga/KD/float16.hpp"
#include "fpga/KD/llapi/zynqmp_api.h"
#include "io/paddle_mobile.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// using namespace cv;

cv::Mat sample_float;

void readImage(std::string filename, float* buffer) {
  std::cout << "readImage1" << std::endl;
  cv::Mat img = cv::imread(filename);
  if (img.empty()) {
    std::cerr << "Can't read image from the file: " << filename << std::endl;
    exit(-1);
  }
  std::cout << "readImage2" << std::endl;
  cv::Mat img2;
  cv::resize(img, img2, cv::Size(224, 224));

  img2.convertTo(sample_float, CV_32FC3);
  std::cout << "readImage3" << sample_float.rows << "," << sample_float.cols
            << std::endl;
  int index = 0;
  for (int row = 0; row < sample_float.rows; ++row) {
    float* ptr = reinterpret_cast<float*>(sample_float.ptr(row));
    for (int col = 0; col < sample_float.cols; col++) {
      float* uc_pixel = ptr;
      // uc_pixel[0] -= 102;
      // uc_pixel[1] -= 117;
      // uc_pixel[1] -= 124;
      float b = uc_pixel[0];
      float g = uc_pixel[1];
      float r = uc_pixel[2];

      buffer[index] = (b - 103.94) * 0.017;
      buffer[index + 1] = (g - 116.78) * 0.017;
      buffer[index + 2] = (r - 123.68) * 0.017;

      ptr += 3;
      index += 3;
    }
  }
}

void readStream(std::string filename, float* buf) {
  std::ifstream in;
  in.open(filename, std::ios::in);
  if (!in.is_open()) {
    std::cout << "open File Failed." << std::endl;
    return;
  }
  string strOne;
  int i = 0;
  while (!in.eof()) {
    in >> buf[i];
    i++;
  }
  in.close();
}

static const char* g_model = "../../../models/easydl_mobilenet_v1";
static const char* image = "../../../models/easydl_mobilenet_v1/4.jpg";

int main() {
  paddle_mobile::zynqmp::open_device();
  paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;
  std::string model = std::string(g_model) + "/model";
  std::string params = std::string(g_model) + "/params";
  if (paddle_mobile.Load(model, params, true)) {
    // if (paddle_mobile.Load(std::string(g_resnet50), false)) {
    Tensor input_tensor;
    SetupTensor<float>(&input_tensor, {1, 3, 224, 224}, static_cast<float>(1),
                       static_cast<float>(1));
    float* data = input_tensor.mutable_data<float>({1, 3, 224, 224});

    // for (int i = 0; i < 3 * 224 * 224; i++) {
    //   data[i] = 1.0f;
    // }
    readImage(image, data);

    auto time3 = time();
    // for (int i = 0; i < 1000; i++) {
      paddle_mobile.Predict(input_tensor);
    // }
    auto time4 = time();
    std::cout << "predict cost: " << time_diff(time3, time4) << "ms\n";

    auto result_ptr = paddle_mobile.Fetch();

    int index = 1;
    float max = 0.0f;
    float* result_data = result_ptr->data<float>();
    for (int i = 0; i < result_ptr->numel(); i++) {
      if (result_data[i] > max) {
        max = result_data[i];
        index = i + 1;
      }
    }
    std::cout << index << "," << max << std::endl;
  }
  paddle_mobile::zynqmp::close_device();
  return 0;
}
