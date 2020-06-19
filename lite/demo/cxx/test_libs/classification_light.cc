// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <fstream>
#include <iostream>
#include "paddle_api.h"   // NOLINT
#include "test_helper.h"  // NOLINT

DEFINE_string(optimized_model_path, "", "the path of optimized model");
DEFINE_string(img_path, "", "the path of input image");
DEFINE_string(img_txt_path,
              "",
              "the path of input image, the image is processed "
              " and saved in txt file");
DEFINE_double(out_max_value, 0.0, "The max value in output tensor");
DEFINE_double(threshold,
              1e-3,
              "If the max value diff is smaller than threshold, pass test");
DEFINE_int32(out_max_value_index, -1, "The max value index in output tensor");

void Run(const std::string& model_path,
         const std::string& img_path,
         const std::string& img_txt_path,
         const float out_max_value,
         const int out_max_value_index,
         const float threshold,
         const int height,
         const int width) {
  // set config and create predictor
  paddle::lite_api::MobileConfig config;
  config.set_threads(3);
  config.set_model_from_file(model_path);

  auto predictor = paddle::lite_api::CreatePaddlePredictor(config);

  // set input
  auto input_tensor = predictor->GetInput(0);
  input_tensor->Resize({1, 3, height, width});
  auto input_data = input_tensor->mutable_data<float>();
  if (img_txt_path.size() > 0) {
    std::fstream fs(img_txt_path);
    if (!fs.is_open()) {
      std::cerr << "Fail to open img txt file:" << img_txt_path << std::endl;
    }
    int num = 1 * 3 * height * width;
    for (int i = 0; i < num; i++) {
      fs >> input_data[i];
    }
  } else {
    cv::Mat img = imread(img_path, cv::IMREAD_COLOR);
    if (!img.data) {
      std::cerr << "Fail to open img:" << img_path << std::endl;
      exit(1);
    }
    float means[3] = {0.485f, 0.456f, 0.406f};
    float scales[3] = {0.229f, 0.224f, 0.225f};
    process_img(img, width, height, input_data, means, scales);
  }

  predictor->Run();

  auto out_tensor = predictor->GetOutput(0);
  auto* out_data = out_tensor->data<float>();
  int64_t output_num = ShapeProduction(out_tensor->shape());
  float max_value = out_data[0];
  int max_index = 0;
  for (int i = 0; i < output_num; i++) {
    if (max_value < out_data[i]) {
      max_value = out_data[i];
      max_index = i;
    }
  }

  std::cout << "max_value:" << max_value << std::endl;
  std::cout << "max_index:" << max_index << std::endl;
  std::cout << "max_value_ground_truth:" << out_max_value << std::endl;
  std::cout << "max_index_ground_truth:" << out_max_value_index << std::endl;
  if (max_index != out_max_value_index ||
      fabs(max_value - out_max_value) > threshold) {
    std::cerr << "----------Fail Test---------- \n\n";
  } else {
    std::cout << "----------Pass Test---------- \n\n";
  }
}

int main(int argc, char** argv) {
  // Check inputs
  google::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_optimized_model_path.empty() ||
      (FLAGS_img_path.empty() && FLAGS_img_txt_path.empty())) {
    std::cerr << "Input error." << std::endl;
    std::cerr
        << "Usage: " << argv[0] << std::endl
        << "--optimized_model_path: the path of optimized model \n"
           "--img_txt_path: the path of input image, the image is processed \n"
           "  and saved in txt file \n"
           "--img_path: the path of input image \n"
           "--out_max_value: The max value in output tensor \n"
           "--threshold: If the max value diff is smaller than threshold,\n"
           "  pass test. Default 1e-3.\n"
           "--out_max_value_index: The max value index in output tensor \n";
    exit(1);
  }

  const int height = 224;
  const int width = 224;
  // Run test
  Run(FLAGS_optimized_model_path,
      FLAGS_img_path,
      FLAGS_img_txt_path,
      FLAGS_out_max_value,
      FLAGS_out_max_value_index,
      FLAGS_threshold,
      height,
      width);
  return 0;
}
