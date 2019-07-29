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
cv::Mat resize_img;

void readImage(std::string filename, float* buffer) {
  std::cout << "readImage1" << std::endl;
  img = imread(filename);
  if (img.empty()) {
    std::cerr << "Can't read image from the file: " << filename << std::endl;
    exit(-1);
  }
  std::cout << "readImage2" << std::endl;
  cv::Mat img2;
  resize(img, img2, Size(608, 608), INTER_AREA);
  resize_img = img2;

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

      buffer[index] = (r - 123.68) * 0.0131;
      buffer[index + 1] = (g - 103.94) * 0.0131;
      buffer[index + 2] = (b - 116.78) * 0.0131;

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

    if (score > 0.4) {
      // int x1 = static_cast<int>(data[2] * mat.cols);
      // int y1 = static_cast<int>(data[3] * mat.rows);
      // int x2 = static_cast<int>(data[4] * mat.cols);
      // int y2 = static_cast<int>(data[5] * mat.rows);
      int x1 = static_cast<int>(data[2]);
      int y1 = static_cast<int>(data[3]);
      int x2 = static_cast<int>(data[4]);
      int y2 = static_cast<int>(data[5]);
      int width = x2 - x1;
      int height = y2 - y1;

      cv::Point pt1(x1, y1);
      cv::Point pt2(x2, y2);
      cv::rectangle(resize_img, pt1, pt2, cv::Scalar(0, 0, 255));
      std::cout << "label:" << index << ",score:" << score << " loc:";
      std::cout << x1 << "," << y1 << "," << width << "," << height
                << std::endl;
    }
    data += 6;
  }
  imwrite("result.jpg", resize_img);
}

PaddleMobileConfig GetConfig() {
    PaddleMobileConfig config;
    config.precision = PaddleMobileConfig::FP32;
    config.device = PaddleMobileConfig::kFPGA;
    // config.model_dir = "../models/mobilenet/";
    config.prog_file = "../../../models/aistudio_attention/model";
    config.param_file = "../../../models/aistudio_attention/params";
    config.thread_num = 1;
    return config;
}

  
  int main() {
    zynqmp::open_device();
    PaddleMobileConfig config = GetConfig();
    config.optimize = true;
 
    auto predictor =
       CreatePaddlePredictor<PaddleMobileConfig,
                             PaddleEngineKind::kPaddleMobile>(config);

   float data[1 * 3 * 48 * 512] = {1.0f};
   memset(data, 1,  (3 * 48 * 512) * sizeof(float));
   for(int i=0; i<3 * 48 * 512; i++) {
        data[i] = 1.0f;
   }
   // readImage("../../../models/easydl_yolov3/DJI_0618.jpeg", data);
 
   PaddleTensor tensor;
   tensor.shape = std::vector<int>({1, 3, 608, 608});
   tensor.data = PaddleBuf(data, sizeof(data));
   tensor.dtype = PaddleDType::FLOAT32;

   // PaddleTensor im_shape_tensor;
   // im_shape_tensor.shape = std::vector<int>({1, 2});
   // float im_shape_data[2] = {608};
   // im_shape_tensor.data = PaddleBuf(im_shape_data, sizeof(im_shape_data));
   // im_shape_tensor.dtype = PaddleDType::FLOAT32;

   std::vector<PaddleTensor> paddle_tensor_feeds(1, tensor);
   // std::vector<PaddleTensor> paddle_tensor_feeds(2);
   // paddle_tensor_feeds.push_back(tensor);
   // paddle_tensor_feeds.push_back(im_shape_tensor);
 

   PaddleTensor tensor_out;
   tensor_out.shape = std::vector<int>({});
   tensor_out.data = PaddleBuf();
   tensor_out.dtype = PaddleDType::FLOAT32;
   std::vector<PaddleTensor> outputs(1, tensor_out);
 
   for (int i = 0; i < 5; ++i) {
       predictor->Run(paddle_tensor_feeds, &outputs);
   
 
     float* data_ret = static_cast<float*>(outputs[0].data.data());
     drawRect(img, data_ret, outputs[0].shape[0]);
  }
   return 0;
}


// static const char* g_inception = "../../../models/yolov3";
// static const char* image = "../../../models/yolov3/1.jpg";

// int main() {
//   paddle_mobile::zynqmp::open_device();
//   paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;
//   std::string model = std::string(g_inception) + "/model";
//   std::string params = std::string(g_inception) + "/params";
//   if (paddle_mobile.Load(model, params, true)) {
//     // if (paddle_mobile.Load(std::string(g_resnet50), false)) {
//     Tensor input_tensor;
//     SetupTensor<float>(&input_tensor, {1, 3, 608, 608}, static_cast<float>(1),
//                        static_cast<float>(1));
//     float* data = input_tensor.mutable_data<float>({1, 3, 608, 608});

//     for (int i = 0; i < 3 * 608 * 608; i++) {
//       data[i] = 1.0f;
//     }
//     // readImage(image, data);

//     auto time3 = time();
//     paddle_mobile.Predict(input_tensor);
//     auto time4 = time();
//     std::cout << "predict cost: " << time_diff(time3, time4) << "ms\n";

//     auto result_ptr = paddle_mobile.Fetch();

//     int index = 1;
//     float* result_data = result_ptr->data<float>();
//     drawRect(img, result_data, result_ptr->dims()[0]);
//   }
//   paddle_mobile::zynqmp::close_device();
//   return 0;
// }
