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
#include "../../src/common/types.h"
#include "../test_helper.h"
#include "../test_include.h"

int main() {
  paddle_mobile::PaddleMobileConfigInternal config;
  config.load_when_predict = true;

#ifdef PADDLE_MOBILE_CL
  paddle_mobile::PaddleMobile<paddle_mobile::GPU_CL> paddle_mobile(config);
  paddle_mobile.SetCLPath("/data/local/tmp/bin");
#else
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
#endif
  //  paddle_mobile.SetThreadNum(4);

  int max = 10;
  auto time1 = paddle_mobile::time();
  auto isok = paddle_mobile.Load(std::string(g_super) + "/model",
                                 std::string(g_super) + "/params", true, false,
                                 1, false);

  if (isok) {
    auto time2 = paddle_mobile::time();
    std::cout << "load cost :" << paddle_mobile::time_diff(time1, time2) << "ms"
              << std::endl;

    // 300 * 300
    std::vector<float> input;
    std::vector<int64_t> dims{1, 1, 300, 300};
    GetInput<float>(g_test_image_1x3x224x224, &input, dims);
    paddle_mobile.Predict(input, dims);

    // 640 * 360 (360P)
    std::vector<float> input1;
    std::vector<int64_t> dims1{1, 1, 640, 360};
    GetInput<float>(g_test_image_1x3x224x224, &input1, dims1);
    auto time3 = paddle_mobile::time();
    for (int i = 0; i < max; ++i) {
      auto time1 = paddle_mobile::time();
      paddle_mobile.Predict(input1, dims1);
      auto time2 = paddle_mobile::time();
      std::cout << "640 * 360 predict cost :第" << i << ": "
                << paddle_mobile::time_diff(time1, time2) << "ms" << std::endl;
    }
    auto time4 = paddle_mobile::time();
    std::cout << "640 * 360 predict cost :"
              << paddle_mobile::time_diff(time3, time4) / max << "ms"
              << std::endl;

    // 720 * 480 (480P)
    std::vector<float> input2;
    std::vector<int64_t> dims2{1, 1, 720, 480};
    GetInput<float>(g_test_image_1x3x224x224, &input2, dims2);
    auto time5 = paddle_mobile::time();
    for (int i = 0; i < max; ++i) {
      auto time1 = paddle_mobile::time();
      paddle_mobile.Predict(input2, dims2);
      auto time2 = paddle_mobile::time();
      std::cout << "720 * 480 predict cost :第" << i << ": "
                << paddle_mobile::time_diff(time1, time2) << "ms" << std::endl;
    }
    auto time6 = paddle_mobile::time();
    std::cout << "720 * 480 predict cost :"
              << paddle_mobile::time_diff(time5, time6) / max << "ms"
              << std::endl;

    // 1024 * 576 (576P)
    std::vector<float> input3;
    std::vector<int64_t> dims3{1, 1, 1024, 576};
    GetInput<float>(g_test_image_1x3x224x224, &input3, dims3);
    auto time7 = paddle_mobile::time();
    for (int i = 0; i < max; ++i) {
      auto time1 = paddle_mobile::time();
      paddle_mobile.Predict(input3, dims3);
      auto time2 = paddle_mobile::time();
      std::cout << "1024 * 576 predict cost :第" << i << ": "
                << paddle_mobile::time_diff(time1, time2) << "ms" << std::endl;
    }
    auto time8 = paddle_mobile::time();
    std::cout << "1024 * 576 predict cost :"
              << paddle_mobile::time_diff(time7, time8) / max << "ms"
              << std::endl;

    // 1280 * 720
    std::vector<float> input4;
    std::vector<int64_t> dims4{1, 1, 1280, 720};
    GetInput<float>(g_test_image_1x3x224x224, &input4, dims4);
    auto time9 = paddle_mobile::time();
    for (int i = 0; i < max; ++i) {
      auto time1 = paddle_mobile::time();
      paddle_mobile.Predict(input4, dims4);
      auto time2 = paddle_mobile::time();
      std::cout << "1280 * 720 predict cost :第" << i << ": "
                << paddle_mobile::time_diff(time1, time2) << "ms" << std::endl;
    }
    auto time10 = paddle_mobile::time();
    std::cout << "1280 * 720 predict cost :"
              << paddle_mobile::time_diff(time9, time10) / max << "ms"
              << std::endl;
  }

  return 0;
}
