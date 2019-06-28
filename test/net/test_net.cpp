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
#include <string>
#include "../test_helper.h"
#include "../test_include.h"

void test(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  test(argc, argv);
  return 0;
}

void test(int argc, char *argv[]) {
  int arg_index = 1;
  bool fuse = std::stoi(argv[arg_index]) == 1;
  arg_index++;
  bool enable_memory_optimization = std::stoi(argv[arg_index]) == 1;
  arg_index++;
  paddle_mobile::PaddleMobileConfigInternal config;
  config.memory_optimization_level = enable_memory_optimization ? MemoryOptimizationWithoutFeeds : NoMemoryOptimization;
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile(config);
  paddle_mobile.SetThreadNum(1);

  int dim_count = std::stoi(argv[arg_index]);
  arg_index++;
  int size = 1;
  std::vector<int64_t> dims;
  for (int i = 0; i < dim_count; i++) {
    int64_t dim = std::stoi(argv[arg_index + i]);
    size *= dim;
    dims.push_back(dim);
  }
  arg_index += dim_count;

  bool is_lod = std::stoi(argv[arg_index]) == 1;
  arg_index++;
  paddle_mobile::framework::LoD lod{{}};
  if (is_lod) {
    int lod_count = std::stoi(argv[arg_index]);
    arg_index++;
    for (int i = 0; i < lod_count; i++) {
      int dim = std::stoi(argv[arg_index + i]);
      lod[0].push_back(dim);
    }
    arg_index += lod_count;
  }

  int var_count = std::stoi(argv[arg_index]);
  arg_index++;
  int sample_step = std::stoi(argv[arg_index]);
  arg_index++;
  std::vector<std::string> var_names;
  for (int i = 0; i < var_count; i++) {
    std::string var_name = argv[arg_index + i];
    var_names.push_back(var_name);
  }
  arg_index += var_count;

  auto time1 = time();
  if (paddle_mobile.Load("./checked_model/model", "./checked_model/params",
                         fuse, false, 1, true)) {
    auto time2 = time();
    std::cout << "auto-test"
              << " load-time-cost :" << time_diff(time1, time2) << "ms"
              << std::endl;

    float input_data_array[size];
    std::ifstream in("input.txt", std::ios::in);
    for (int i = 0; i < size; i++) {
      float num;
      in >> num;
      input_data_array[i] = num;
    }
    in.close();

    auto time3 = time();
    // std::vector<float> input_data;
    // for (int i = 0; i < size; i++) {
    //   float num = input_data_array[i];
    //   input_data.push_back(num);
    // }
    // paddle_mobile::framework::Tensor input_tensor(input_data, paddle_mobile::framework::make_ddim(dims));
    paddle_mobile::framework::Tensor input_tensor(input_data_array, paddle_mobile::framework::make_ddim(dims));
    auto time4 = time();
    std::cout << "auto-test"
              << " preprocess-time-cost :" << time_diff(time3, time4) << "ms"
              << std::endl;

    paddle_mobile::framework::LoDTensor input_lod_tensor;
    if (is_lod) {
      input_lod_tensor.Resize(paddle_mobile::framework::make_ddim(dims));
      input_lod_tensor.set_lod(lod);
      auto *tensor_data = input_lod_tensor.mutable_data<float>();
      for (int i = 0; i < size; i++) {
        tensor_data[i] = input_data_array[i];
      }
    }

    // 预热10次
    for (int i = 0; i < 10; i++) {
      if (is_lod) {
        auto out = paddle_mobile.Predict(input_lod_tensor);
      } else {
        paddle_mobile.Feed(var_names[0], input_tensor);
        paddle_mobile.Predict();
      }
    }

    // 测速
    auto time5 = time();
    for (int i = 0; i < 50; i++) {
      if (is_lod) {
        auto out = paddle_mobile.Predict(input_lod_tensor);
      } else {
        paddle_mobile.Feed(var_names[0], input_tensor);
        paddle_mobile.Predict();
      }
    }
    auto time6 = time();
    std::cout << "auto-test"
              << " predict-time-cost " << time_diff(time5, time6) / 50 << "ms"
              << std::endl;

    // 测试正确性
    if (is_lod) {
      auto out = paddle_mobile.Predict(input_lod_tensor);
    } else {
      paddle_mobile.Feed(var_names[0], input_tensor);
      paddle_mobile.Predict();
    }
    for (auto var_name : var_names) {
      auto out = paddle_mobile.Fetch(var_name);
      auto len = out->numel();
      if (len == 0) {
        continue;
      }
      if (out->memory_size() == 0) {
        continue;
      }
      auto data = out->data<float>();
      std::string sample = "";
      for (int i = 0; i < len; i += sample_step) {
        sample += " " + std::to_string(data[i]);
      }
      std::cout << "auto-test"
                << " var " << var_name << sample << std::endl;
    }
    std::cout << std::endl;
  }
}
