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
  std::vector<int64_t> dims{1, 8, 32, 32};
  int op_index = 2;
  std::string input_var_name = "ConvNdBackward2.conv2d.output.1.tmp_0";
  std::vector<std::string> output_var_names{
      "ConvNdBackward2.conv2d.output.1.tmp_1"};

  bool fuse = false;
  bool enable_memory_optimization = true;
  paddle_mobile::PaddleMobileConfigInternal config;
  config.memory_optimization_level = enable_memory_optimization
                                         ? MemoryOptimizationWithoutFeeds
                                         : NoMemoryOptimization;
#ifdef PADDLE_MOBILE_CL
  // config.load_when_predict = true;
  paddle_mobile::PaddleMobile<paddle_mobile::GPU_CL> paddle_mobile(config);
  paddle_mobile.SetCLPath("/data/local/tmp/bin");
#else
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile(config);
  paddle_mobile.SetThreadNum(1);
#endif

  int size = 1;
  for (int i = 0; i < dims.size(); i++) {
    size *= dims[i];
  }

  bool is_sample_step = false;
  int sample_step = 1;
  int sample_num = 20;

  auto time1 = time();
  if (paddle_mobile.Load("./checked_model/model", "./checked_model/params",
                         fuse, false, 1, true, 1)) {
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
    std::vector<float> input_data;
    for (int i = 0; i < size; i++) {
      float num = input_data_array[i];
      input_data.push_back(num);
    }
    paddle_mobile::framework::Tensor input_tensor(
        input_data, paddle_mobile::framework::make_ddim(dims));
    auto time4 = time();
    std::cout << "auto-test"
              << " preprocess-time-cost :" << time_diff(time3, time4) << "ms"
              << std::endl;

    // 测试正确性
    // 以下代码依赖paddle_mobile.h及executor.h的属性可见性，如需使用，调整可见性后，放开注释
    // auto *input_var =
    //     paddle_mobile.executor_->program_.scope->FindVar(input_var_name);
    // framework::LoDTensor *target =
    //     input_var->template GetMutable<framework::LoDTensor>();
    // target->Resize(input_tensor.dims());
    // target->ShareDataWith(input_tensor);
    // paddle_mobile.executor_->ops_of_block0_[op_index]->InferShape();
    // paddle_mobile.executor_->ops_of_block0_[op_index]->Run();

    for (auto var_name : output_var_names) {
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
      if (!is_sample_step) {
        sample_step = len / sample_num;
      }
      if (sample_step <= 0) {
        sample_step = 1;
      }
      for (int i = 0; i < len; i += sample_step) {
        sample += " " + std::to_string(data[i]);
      }
      std::cout << "auto-test"
                << " var " << var_name << sample << std::endl;
    }
    std::cout << std::endl;
  }
}
