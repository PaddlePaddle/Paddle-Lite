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

void test(int argc, char *argv[], bool fuse);

int main(int argc, char *argv[]) {
  test(argc, argv, false);
  test(argc, argv, true);
  return 0;
}
void test(int argc, char *argv[], bool fuse) {
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  paddle_mobile.SetThreadNum(1);
  std::string tag = fuse ? "-fuse" : "";

  int dim_count = std::stoi(argv[1]);
  int size = 1;
  std::vector<int64_t> dims;
  for (int i = 0; i < dim_count; i++) {
    int64_t dim = std::stoi(argv[2 + i]);
    size *= dim;
    dims.push_back(dim);
  }

  int var_count = std::stoi(argv[1 + dim_count]);
  std::vector<std::string> var_names;
  for (int i = 0; i < var_count; i++) {
    std::string var_name = argv[1 + dim_count + 1 + 1 + i];
    var_names.push_back(var_name);
  }

  auto time1 = time();
  if (paddle_mobile.Load("./checked_model/model", "./checked_model/params",
                         fuse, false, 1, true)) {
    auto time2 = time();
    std::cout << "auto-test" << tag
              << " load-time-cost :" << time_diff(time1, time1) << "ms"
              << std::endl;

    std::vector<float> input_data;
    std::ifstream in("input.txt", std::ios::in);
    for (int i = 0; i < size; i++) {
      float num;
      in >> num;
      input_data.push_back(num);
    }
    in.close();

    // 预热10次
    for (int i = 0; i < 10; i++) {
      auto out = paddle_mobile.Predict(input_data, dims);
    }

    // 测速
    auto time3 = time();
    for (int i = 0; i < 50; i++) {
      auto out = paddle_mobile.Predict(input_data, dims);
    }
    auto time4 = time();
    std::cout << "auto-test" << tag << " predict-time-cost "
              << time_diff(time3, time4) / 50 << "ms" << std::endl;

    // 测试正确性
    auto out = paddle_mobile.Predict(input_data, dims);
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
      int step = len / 20;
      std::string sample = "";
      for (int i = 0; i < len; i += step) {
        sample += " " + std::to_string(data[i]);
      }
      std::cout << "auto-test" << tag << " var " << var_name << sample
                << std::endl;
    }
    std::cout << std::endl;
  }
}
