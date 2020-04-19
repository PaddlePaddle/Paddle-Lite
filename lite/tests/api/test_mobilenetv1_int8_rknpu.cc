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

#include <sys/time.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

inline int64_t ShapeProduction(std::vector<int64_t> shape) {
  int64_t s = 1;
  for (int64_t dim : shape) {
    s *= dim;
  }
  return s;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "[ERROR] usage: ./" << argv[0]
              << " model_dir [thread_num] [warmup_times] [repeat_times] "
                 "[input_data_path] [output_data_path]"
              << std::endl;
    return -1;
  }
  std::string model_dir = argv[1];
  int thread_num = 1;
  if (argc > 2) {
    thread_num = atoi(argv[2]);
  }
  int warmup_times = 5;
  if (argc > 3) {
    warmup_times = atoi(argv[3]);
  }
  int repeat_times = 10;
  if (argc > 4) {
    repeat_times = atoi(argv[4]);
  }
  std::string input_data_path;
  if (argc > 5) {
    input_data_path = argv[5];
  }
  std::string output_data_path;
  if (argc > 6) {
    output_data_path = argv[6];
  }
  paddle::lite_api::CxxConfig config;
  config.set_model_dir(model_dir);
  config.set_threads(thread_num);
  config.set_power_mode(paddle::lite_api::LITE_POWER_HIGH);
  config.set_valid_places(
      {paddle::lite_api::Place{
           TARGET(kARM), PRECISION(kFloat), DATALAYOUT(kNCHW)},
       paddle::lite_api::Place{
           TARGET(kARM), PRECISION(kInt8), DATALAYOUT(kNCHW)},
       paddle::lite_api::Place{
           TARGET(kARM), PRECISION(kInt8), DATALAYOUT(kNCHW)},
       paddle::lite_api::Place{
           TARGET(kRKNPU), PRECISION(kInt8), DATALAYOUT(kNCHW)}});
  auto predictor = paddle::lite_api::CreatePaddlePredictor(config);

  std::unique_ptr<paddle::lite_api::Tensor> input_tensor(
      std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 3, 224, 224});
  auto input_data = input_tensor->mutable_data<float>();
  auto input_size = ShapeProduction(input_tensor->shape());
  if (input_data_path.empty()) {
    for (int i = 0; i < input_size; i++) {
      input_data[i] = 1;
    }
  } else {
    std::fstream fs(input_data_path, std::ios::in);
    if (!fs.is_open()) {
      std::cerr << "open input data file failed." << std::endl;
      return -1;
    }
    for (int i = 0; i < input_size; i++) {
      fs >> input_data[i];
    }
  }

  for (int i = 0; i < warmup_times; ++i) {
    predictor->Run();
  }

  auto start = GetCurrentUS();
  for (int i = 0; i < repeat_times; ++i) {
    predictor->Run();
  }

  std::cout << "Model: " << model_dir << ", threads num " << thread_num
            << ", warmup times: " << warmup_times
            << ", repeat times: " << repeat_times << ", spend "
            << (GetCurrentUS() - start) / repeat_times / 1000.0
            << " ms in average." << std::endl;

  std::unique_ptr<const paddle::lite_api::Tensor> output_tensor(
      std::move(predictor->GetOutput(0)));
  auto output_data = output_tensor->data<float>();
  auto output_size = ShapeProduction(output_tensor->shape());
  std::cout << "output data:";
  for (int i = 0; i < output_size; i += 100) {
    std::cout << "[" << i << "] " << output_data[i] << std::endl;
  }
  return 0;
}
