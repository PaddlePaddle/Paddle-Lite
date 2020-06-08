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

#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "lite/api/paddle_api.h"
#include "lite/api/paddle_use_kernels.h"
#include "lite/api/paddle_use_ops.h"
#include "lite/api/paddle_use_passes.h"
using namespace paddle::lite_api;  // NOLINT

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
           TARGET(kAPU), PRECISION(kInt8), DATALAYOUT(kNCHW)}});
  auto predictor = paddle::lite_api::CreatePaddlePredictor(config);

  std::unique_ptr<paddle::lite_api::Tensor> input_tensor(
      std::move(predictor->GetInput(0)));
  input_tensor->Resize({1, 3, 224, 224});
  auto input_data = input_tensor->mutable_data<float>();
  auto input_size = ShapeProduction(input_tensor->shape());

  // test loop
  int total_imgs = 500;
  float test_num = 0;
  float top1_num = 0;
  float top5_num = 0;
  int output_len = 1000;
  std::vector<int> index(1000);
  bool debug = true;  // false;
  int show_step = 500;
  for (int i = 0; i < total_imgs; i++) {
    // set input
    std::string filename = input_data_path + "/" + std::to_string(i);
    std::ifstream fs(filename, std::ifstream::binary);
    if (!fs.is_open()) {
      std::cout << "open input file fail.";
    }
    auto input_data_tmp = input_data;
    for (int i = 0; i < input_size; ++i) {
      fs.read(reinterpret_cast<char*>(input_data_tmp), sizeof(*input_data_tmp));
      input_data_tmp++;
    }
    int label = 0;
    fs.read(reinterpret_cast<char*>(&label), sizeof(label));
    fs.close();

    if (debug && i % show_step == 0) {
      std::cout << "input data:" << std::endl;
      std::cout << input_data[0] << " " << input_data[10] << " "
                << input_data[input_size - 1] << std::endl;
      std::cout << "label:" << label << std::endl;
    }

    // run
    predictor->Run();
    auto output0 = predictor->GetOutput(0);
    auto output0_data = output0->data<float>();

    // get output
    std::iota(index.begin(), index.end(), 0);
    std::stable_sort(
        index.begin(), index.end(), [output0_data](size_t i1, size_t i2) {
          return output0_data[i1] > output0_data[i2];
        });
    test_num++;
    if (label == index[0]) {
      top1_num++;
    }
    for (int i = 0; i < 5; i++) {
      if (label == index[i]) {
        top5_num++;
      }
    }

    if (debug && i % show_step == 0) {
      std::cout << index[0] << " " << index[1] << " " << index[2] << " "
                << index[3] << " " << index[4] << std::endl;
      std::cout << output0_data[index[0]] << " " << output0_data[index[1]]
                << " " << output0_data[index[2]] << " "
                << output0_data[index[3]] << " " << output0_data[index[4]]
                << std::endl;
      std::cout << output0_data[630] << std::endl;
    }
    if (i % show_step == 0) {
      std::cout << "step " << i << "; top1 acc:" << top1_num / test_num
                << "; top5 acc:" << top5_num / test_num << std::endl;
    }
  }
  std::cout << "final result:" << std::endl;
  std::cout << "top1 acc:" << top1_num / test_num << std::endl;
  std::cout << "top5 acc:" << top5_num / test_num << std::endl;
  return 0;
}
