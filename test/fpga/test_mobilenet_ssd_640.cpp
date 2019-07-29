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

#include "fpga/KD/float16.hpp"
#include "fpga/KD/llapi/zynqmp_api.h"
#include "io/paddle_mobile.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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

  resize(img, img2, Size(640, 640), INTER_AREA);

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

      buffer[index] = (r - 127.5) / 128;
      buffer[index + 1] = (g - 127.5) / 128;
      buffer[index + 2] = (b - 127.5) / 128;

      ptr += 3;
      index += 3;
    }
  }
  // return sample_float;
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

void drawRect(const Mat& mat, float* data, int len) {
  for (int i = 0; i < len; i++) {
    float index = data[0];
    float score = data[1];

    if (score > 0.2) {
      int x1 = static_cast<int>(data[2] * mat.cols);
      int y1 = static_cast<int>(data[3] * mat.rows);
      int x2 = static_cast<int>(data[4] * mat.cols);
      int y2 = static_cast<int>(data[5] * mat.rows);
      int width = x2 - x1;
      int height = y2 - y1;

      cv::Point pt1(x1, y1);
      cv::Point pt2(x2, y2);
      cv::rectangle(mat, pt1, pt2, cv::Scalar(0, 0, 255));
      std::cout << "label:" << index << ",score:" << score << " loc:";
      std::cout << x1 << "," << y1 << "," << width << "," << height
                << std::endl;
    }
    data += 6;
  }
  imwrite("result.jpg", mat);
}

static const char* g_inception = "../../../models/mobilenet_ssd_640";
static const char* image = "../../../models/mobilenet_ssd_640/1.jpeg";

int main() {
  paddle_mobile::zynqmp::open_device();
  paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;
  std::string model = std::string(g_inception) + "/model";
  std::string params = std::string(g_inception) + "/params";
  if (paddle_mobile.Load(model, params, true)) {
    // if (paddle_mobile.Load(std::string(g_resnet50), false)) {
    Tensor input_tensor;
    SetupTensor<float>(&input_tensor, {1, 3, 640, 640}, static_cast<float>(1),
                       static_cast<float>(1));
    float* data = input_tensor.mutable_data<float>({1, 3, 640, 640});

    // for (int i = 0; i < 3 * 300 * 300; i++) {
    //   data[i] = 1.0f;
    // }
    readImage(image, data);

    auto time3 = time();
    // for (int i = 0; i < 1000; i++) {
      paddle_mobile.Predict(input_tensor);
    // }
    auto time4 = time();
    std::cout << "predict 1000 cost: " << time_diff(time3, time4) / 100 << "ms\n";

    auto result_ptr = paddle_mobile.Fetch();

    int index = 1;
    float* result_data = result_ptr->data<float>();
    drawRect(img, result_data, result_ptr->dims()[0]);
  }
  paddle_mobile::zynqmp::close_device();
  return 0;
}
