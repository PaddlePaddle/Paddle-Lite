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
#include "../test_include.h"

#include "fpga/KD/float16.hpp"
#include "fpga/KD/llapi/zynqmp_api.h"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace paddle_mobile;
using namespace cv;

cv::Mat sample_float;

int width = 300;
int height = 300;

static size_t ReadBuffer(const char *file_name, uint8_t **out) {
  FILE *fp;
  fp = fopen(file_name, "rb");
  // PADDLE_MOBILE_ENFORCE(fp != NULL, " %s open failed !", file_name);

  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  rewind(fp);

  *out = reinterpret_cast<uint8_t *>(malloc(size));

  size_t cur_len = 0;
  size_t nread;
  while ((nread = fread(*out + cur_len, 1, size - cur_len, fp)) != 0) {
    cur_len += nread;
  }
  fclose(fp);
  return cur_len;
}

void readImage(std::string filename, float *buffer) {
  Mat img = imread(filename);
  if (img.empty()) {
    std::cerr << "Can't read image from the file: " << filename << std::endl;
    exit(-1);
  }

  Mat img2;
  resize(img, img2, Size(width, height));

  img2.convertTo(sample_float, CV_32FC3);

  int index = 0;
  for (int row = 0; row < sample_float.rows; ++row) {
    float *ptr = reinterpret_cast<float *>(sample_float.ptr(row));
    for (int col = 0; col < sample_float.cols; col++) {
      float *uc_pixel = ptr;

      float r = uc_pixel[0];
      float g = uc_pixel[1];
      float b = uc_pixel[2];

      buffer[index] = b - 104;
      buffer[index + 1] = g - 117;
      buffer[index + 2] = r - 124;

      // float b = uc_pixel[0];
      // float g = uc_pixel[1];
      // float r = uc_pixel[2];

      // buffer[index] = (b - 128) / 128;
      // buffer[index + 1] = (g - 128) / 128;
      // buffer[index + 2] = (r - 128) / 128;

      ptr += 3;
      index += 3;
    }
  }
}

void drawRect(const Mat &mat, float *data, int len) {
  for (int i = 0; i < len; i++) {
    float index = data[0];
    float score = data[1];
    if (score > 0.5) {
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

PaddleMobileConfig GetConfig() {
  PaddleMobileConfig config;
  config.precision = PaddleMobileConfig::FP32;
  config.device = PaddleMobileConfig::kFPGA;
  // config.model_dir = "../models/mobilenet/";
  config.prog_file = "../models/vgg-ssd/model";
  config.param_file = "../models/vgg-ssd/params";
  config.thread_num = 4;
  return config;
}

int main() {
  zynqmp::open_device();
  PaddleMobileConfig config = GetConfig();

  const auto &memory_pack = std::make_shared<PaddleModelMemoryPack>();

  uint8_t *model = nullptr;
  size_t model_size = ReadBuffer(config.prog_file.c_str(), &model);
  uint8_t *params = nullptr;
  size_t param_size = ReadBuffer(config.param_file.c_str(), &params);

  memory_pack->model_size = model_size;
  memory_pack->model_buf = model;
  memory_pack->combined_params_size = param_size;
  memory_pack->combined_params_buf = params;
  memory_pack->from_memory = true;
  config.precision = PaddleMobileConfig::FP32;
  config.device = PaddleMobileConfig::kFPGA;
  config.memory_pack = *memory_pack;
  config.thread_num = 4;
  config.optimize = true;

  auto predictor =
      CreatePaddlePredictor<PaddleMobileConfig,
                            PaddleEngineKind::kPaddleMobile>(config);

  float data[1 * 3 * width * height] = {1.0f};
  readImage("1.jpg", data);

  PaddleTensor tensor;
  tensor.shape = std::vector<int>({1, 3, width, height});
  tensor.data = PaddleBuf(data, sizeof(data));
  tensor.dtype = PaddleDType::FLOAT32;
  std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);

  PaddleTensor tensor_out;
  tensor_out.shape = std::vector<int>({});
  tensor_out.data = PaddleBuf();
  tensor_out.dtype = PaddleDType::FLOAT32;
  std::vector<PaddleTensor> outputs(1, tensor_out);

  std::cout << " before predict " << std::endl;
  auto time3 = time();
  // for (int i=0; i<1000; i++) {
    predictor->Run(paddle_tensor_feeds, &outputs);
  // }  
  auto time4 = time();  
  std::cout << "predict 1000 cost: " << time_diff(time3, time4) << "ms\n";

  std::cout << " after predict " << std::endl;
  //  assert();

  float *data_o = static_cast<float *>(outputs[0].data.data());
  for (size_t j = 0; j < outputs[0].data.length() / sizeof(float); ++j) {
    // std::cout << "output[" << j << "]: " << data_o[j] << std::endl;
  }

  drawRect(sample_float, data_o, 20);

  return 0;
}
