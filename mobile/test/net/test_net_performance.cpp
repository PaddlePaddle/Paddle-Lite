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

#include <unistd.h>
#include <fstream>
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
  bool quantification = std::stoi(argv[arg_index]) == 1;
  arg_index++;
  int quantification_fold = std::stoi(argv[arg_index]);
  arg_index++;
  paddle_mobile::PaddleMobileConfigInternal config;
  config.memory_optimization_level = enable_memory_optimization
                                         ? MemoryOptimizationWithoutFeeds
                                         : NoMemoryOptimization;

  // save obfuscated model
  // config.model_obfuscate_key = "asdf";
  // std::ofstream out_file("new-params", std::ofstream::binary);
  // char *out_data = ReadFileToBuff("./checked_model/params");
  // int len = GetFileLength("./checked_model/params");
  // out_file.write(out_data, len);
  // out_file.close();

#ifdef PADDLE_MOBILE_CL
  //  config.load_when_predict = true;
  paddle_mobile::PaddleMobile<paddle_mobile::GPU_CL> paddle_mobile(config);
  paddle_mobile.SetCLPath("/data/local/tmp/bin");
  std::cout << "testing opencl performance " << std::endl;
#else
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile(config);
  paddle_mobile.SetThreadNum(1);
  std::cout << "testing cpu performance " << std::endl;
#endif

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
  bool is_sample_step = std::stoi(argv[arg_index]) == 1;
  arg_index++;
  int sample_arg = std::stoi(argv[arg_index]);
  int sample_step = sample_arg;
  int sample_num = sample_arg;
  arg_index++;
  std::vector<std::string> var_names;
  for (int i = 0; i < var_count; i++) {
    std::string var_name = argv[arg_index + i];
    var_names.push_back(var_name);
  }
  arg_index += var_count;
  bool check_shape = std::stoi(argv[arg_index]) == 1;
  arg_index++;

  int run_times = std::stoi(argv[arg_index]);
  arg_index++;

  bool warm_up = std::stoi(argv[arg_index]) == 1;
  arg_index++;

  auto time1 = time();
  if (paddle_mobile.Load("./checked_model/model", "./checked_model/params",
                         fuse, quantification, 1, is_lod,
                         quantification_fold)) {
    auto time2 = time();
    std::cout << "auto-test"
              << " load-time-cost :" << time_diff(time1, time2) << "ms"
              << std::endl;

    float *input_data_array = new float[size];
    std::ifstream in("input.txt", std::ios::in);
    for (int i = 0; i < size; i++) {
      float num;
      in >> num;
      input_data_array[i] = num;
    }
    in.close();

    auto time3 = time();

    paddle_mobile::framework::Tensor input_tensor(
        input_data_array, paddle_mobile::framework::make_ddim(dims));
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
    if (warm_up) {
      for (int i = 0; i < 10; i++) {
        if (is_lod) {
          auto out = paddle_mobile.Predict(input_lod_tensor);
        } else {
          paddle_mobile.Feed(var_names[0], input_tensor);
          paddle_mobile.Predict();
        }
      }
    }

    // 测速
    auto max_time = -1;
    auto min_time = 100000;
    auto all_time = 0;
    if (is_lod) {
      for (int i = 0; i < run_times; i++) {
        auto time7 = time();
        paddle_mobile.Predict(input_lod_tensor);
        auto time8 = time();
        const double diff_time_single = time_diff(time7, time8);
        max_time = fmax(diff_time_single, max_time);
        min_time = fmin(diff_time_single, min_time);
        all_time += diff_time_single;
      }
    } else {
      paddle_mobile.Feed(var_names[0], input_tensor);
      for (int i = 0; i < run_times; i++) {
        auto time7 = time();
        paddle_mobile.Predict();
        auto time8 = time();
        usleep(1000 * quantification_fold);
        const double diff_time_single = time_diff(time7, time8);
        max_time = fmax(diff_time_single, max_time);
        min_time = fmin(diff_time_single, min_time);
        all_time += diff_time_single;
      }
    }

    std::cout << "auto-test"
              << " predict-time-cost-avg " << all_time * 1.0f / run_times
              << "ms" << std::endl;
    std::cout << "auto-test"
              << " predict-time-cost-max " << double(max_time) << "ms"
              << std::endl;
    std::cout << "auto-test"
              << " predict-time-cost-min " << double(min_time) << "ms"
              << std::endl;

    std::cout << std::endl;
  }
}
