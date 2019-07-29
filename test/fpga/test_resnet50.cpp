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

using namespace paddle_mobile;
using namespace cv;

cv::Mat img;
cv::Mat sample_float;

void readImage(std::string filename, float* buffer) {
  std::cout << "readImage1" << std::endl;
  img = imread(filename);
  if (img.empty()) {
    std::cerr << "Can't read image from the file: " << filename << std::endl;
    exit(-1);
  }
  std::cout << "readImage2" << std::endl;
  cv::Mat img2;

  int channel = img.channels();
  int width = img.cols;
  int height = img.rows;
  std::cout << "mat:" << width << "," << height << "," << channel << std::endl;

  resize(img, img2, Size(224, 224), INTER_AREA);

  img2.convertTo(sample_float, CV_32FC3);

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

      buffer[index] = (r / 255 - 0.485) / 0.229;
      buffer[index + 1] = (g / 255 - 0.456) * 0.224;
      buffer[index + 2] = (b / 255  - 0.406) * 0.225;

      ptr += 3;
      index += 3;
    }
  }
  // return sample_float;
}

static const char *g_resnet50 = "../../../models/resnet50";
const std::string image = "../../../models/resnet50/3.jpg";  // NOLINT
int main() {
  paddle_mobile::zynqmp::open_device();
  paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;
  std::string model = std::string(g_resnet50) + "/model";
  std::string params = std::string(g_resnet50) + "/params";
 if (paddle_mobile.Load(model, params, true)) {
    Tensor input_tensor;

    SetupTensor<float>(&input_tensor, {1, 3, 224, 224}, static_cast<float>(1),
                       static_cast<float>(1));
    float* data = input_tensor.mutable_data<float>({1, 3, 224, 224});

    readImage(image, data);
    // for (int i = 0; i < 3*224*224; ++i)
    // {
    //   data[i] == 1;
    // }

    auto time3 = time();
    int times = 1000;
    // for (int i = 0; i < times; i++) {
      paddle_mobile.Predict(input_tensor);
    // }
    auto time4 = time();
    std::cout << "predict cost: " << time_diff(time3, time4) / times << "ms\n";

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
