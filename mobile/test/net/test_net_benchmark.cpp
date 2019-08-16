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
  paddle_mobile.SetThreadNum(1);
  auto time1 = paddle_mobile::time();

  auto isok =
      paddle_mobile.Load(std::string(g_yolo) + "/model",
                         std::string(g_yolo) + "/params", true, false, 1, true);
  if (isok) {
    auto time2 = paddle_mobile::time();
    std::cout << "load cost :" << paddle_mobile::time_diff(time1, time1) << "ms"
              << std::endl;

    std::vector<float> input;
    std::vector<int64_t> dims{1, 3, 64, 64};
    GetInput<float>(g_test_image_1x3x224x224_banana, &input, dims);

    paddle_mobile::framework::DDim ddim =
        paddle_mobile::framework::make_ddim(dims);
    Tensor feed_tensor(input, paddle_mobile::framework::make_ddim(dims));

    // 预热十次
    for (int i = 0; i < 10; ++i) {
      //      auto vec_result = paddle_mobile.Predict(input, dims);
      paddle_mobile.Feed("data", feed_tensor);
      paddle_mobile.Predict();
    }
    auto time3 = paddle_mobile::time();
    for (int i = 0; i < 100; ++i) {
      //      auto vec_result = paddle_mobile.Predict(input, dims);
      paddle_mobile.Feed("data", feed_tensor);
      paddle_mobile.Predict();
    }
    auto time4 = paddle_mobile::time();
    std::cout << "predict cost :"
              << paddle_mobile::time_diff(time3, time4) / 100 << "ms"
              << std::endl;
  }

  return 0;
}
