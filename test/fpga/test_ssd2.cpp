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
#include "io/paddle_inference_api.h"

#include "fpga/KD/float16.hpp"
#include "fpga/KD/llapi/zynqmp_api.h"

// using namespace paddle_mobile;

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
// using namespace cv;

cv::Mat sample_float;

void readImage(std::string filename, float* buffer) {
  Mat img = imread(filename);
  if (img.empty()) {
    std::cerr << "Can't read image from the file: " << filename << std::endl;
    exit(-1);
  }

  Mat img2;
  resize(img, img2, Size(300, 300));

  img2.convertTo(sample_float, CV_32FC3);

  int index = 0;
  for (int row = 0; row < sample_float.rows; ++row) {
    float* ptr = reinterpret_cast<float*>(sample_float.ptr(row));
    for (int col = 0; col < sample_float.cols; col++) {
      float* uc_pixel = ptr;
      float r = uc_pixel[0];
      float g = uc_pixel[1];
      float b = uc_pixel[2];

      buffer[index] = b - 104;
      buffer[index + 1] = g - 117;
      buffer[index + 2] = r - 124;
      ptr += 3;
      index += 3;
    }
  }
}

void drawRect(const Mat& mat, float* data, int len) {
  for (int i = 0; i < len; i++) {
    float index = data[0];
    float score = data[1];
    std::cout << index << " score::" << score << std::endl;

    if (score > 0.5) {
      float x0 = data[2] * 300;
      float y0 = data[3] * 300;
      float x1 = data[4] * 300;
      float y1 = data[5] * 300;

      cv::Point pt1(x0, y0);
      cv::Point pt2(x1, y1);
      cv::rectangle(mat, pt1, pt2, cv::Scalar(0, 0, 255));
    }
    std::cout << "score::" << score;
    // std::cout
    data += 6;
  }
  imwrite("result.jpg", mat);
}

PaddleMobileConfig GetConfig() {
  PaddleMobileConfig config;
  config.precision = PaddleMobileConfig::FP32;
  config.device = PaddleMobileConfig::kFPGA;
  // config.model_dir = "../models/mobilenet/";
  config.prog_file = "../models/ssd/model";
  config.param_file = "../models/ssd/params";
  config.thread_num = 4;
  return config;
}

int main() {
  zynqmp::open_device();
  PaddleMobileConfig config = GetConfig();
  auto predictor =
      CreatePaddlePredictor<PaddleMobileConfig,
                            PaddleEngineKind::kPaddleMobile>(config);

  float data[1 * 3 * 300 * 300] = {1.0f};
  readImage("1.jpg", data);

  PaddleTensor tensor;
  tensor.shape = std::vector<int>({1, 3, 300, 300});
  tensor.data = PaddleBuf(data, sizeof(data));
  tensor.dtype = PaddleDType::FLOAT32;
  std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);

  PaddleTensor tensor_out;
  tensor_out.shape = std::vector<int>({});
  tensor_out.data = PaddleBuf();
  tensor_out.dtype = PaddleDType::FLOAT32;
  std::vector<PaddleTensor> outputs(1, tensor_out);

  std::cout << " before predict " << std::endl;

  predictor->Run(paddle_tensor_feeds, &outputs);

  std::cout << " after predict " << std::endl;
  //  assert();

  float* data_o = static_cast<float*>(outputs[0].data.data());
  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); ++j) {
    // std::cout << "output[" << j << "]: " << data_o[j] << std::endl;
  }

  drawRect(sample_float, data_o, 20);

  return 0;
}
