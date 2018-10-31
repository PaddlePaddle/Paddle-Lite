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

int main() {
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  paddle_mobile.SetThreadNum(4);
  auto time1 = time();
  auto isok = paddle_mobile.Load(
      std::string(g_mobilenet_ssd_gesture) + "/model",
      std::string(g_mobilenet_ssd_gesture) + "/params", true);
  //  auto isok = paddle_mobile.Load(g_mobilenet_ssd, false);
  if (isok) {
    auto time2 = time();
    std::cout << "load cost :" << time_diff(time1, time2) << "ms" << std::endl;

    std::vector<float> input;
    std::vector<int64_t> dims{1, 3, 300, 300};
    GetInput<float>(g_hand, &input, dims);

    // 预热十次
    for (int i = 0; i < 10; ++i) {
      auto output = paddle_mobile.Predict(input, dims);
    }
    auto time3 = time();
    for (int i = 0; i < 10; ++i) {
      auto output = paddle_mobile.Predict(input, dims);
    }
    auto time4 = time();
    std::cout << "predict cost :" << time_diff(time3, time4) / 10 << "ms"
              << std::endl;
  }
  return 0;
}
