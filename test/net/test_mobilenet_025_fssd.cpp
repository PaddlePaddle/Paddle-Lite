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
#include "../test_helper.h"
#include "../test_include.h"

int main(int argc, char **argv) {
  int times = 10;
  if (argc <= 1) {
    times = 10;
    std::cout << "没有输入 , 使用默认10次 " << times << std::endl;
  } else {
    std::string arstr = argv[1];
    times = std::stoi(arstr);
    std::cout << "input times: " << times << std::endl;
  }

  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  paddle_mobile.SetThreadNum(1);
  auto isok =
      paddle_mobile.Load(std::string(g_fluid_fssd_new) + "/model",
                         std::string(g_fluid_fssd_new) + "/params", true);
  if (isok) {
    std::vector<float> input;
    std::vector<int64_t> dims{1, 3, 160, 160};
    GetInput<float>(g_imgfssd_ar1, &input, dims);
    std::cout << "预热10次....." << std::endl;

    // 预热十次
    for (int i = 0; i < 10; ++i) {
      auto output = paddle_mobile.Predict(input, dims);
    }
    std::cout << "开始....." << std::endl;

    double time_sum = 0;

    for (int i = 0; i < times; ++i) {
      auto time3 = time();
      auto output = paddle_mobile.Predict(input, dims);
      auto time4 = time();
      double timeDiff = time_diff(time3, time4);
      time_sum += timeDiff;
      std::cout << "第" << i << "次"
                << "predict cost :" << timeDiff << "ms" << std::endl;
    }
    std::cout << "平均时间:" << time_sum / times << "ms" << std::endl;
  }
  return 0;
}
