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

  auto time1 = paddle_mobile::time();
#ifdef PADDLE_MOBILE_CL
  paddle_mobile::PaddleMobile<paddle_mobile::GPU_CL> paddle_mobile(config);
  paddle_mobile.SetCLPath("/data/local/tmp/bin");
#else
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile(config);
#endif
  //  paddle_mobile.SetThreadNum(4);

  auto isok = paddle_mobile.Load(std::string(g_super) + "/model",
                                 std::string(g_super) + "/params", true, false,
                                 1, false);

  //  auto isok = paddle_mobile.Load(std::string(g_mobilenet_mul), true);
  if (isok) {
    auto time2 = paddle_mobile::time();
    std::cout << "load cost :" << paddle_mobile::time_diff(time1, time2) << "ms"
              << std::endl;
    // 300*300
    //    std::vector<float> input;
    //    std::vector<int64_t> dims{1, 1, 300, 300};
    //    GetInput<float>(g_test_image_1x3x224x224, &input, dims);
    //
    //    std::vector<float> vec_result;

    auto time3 = paddle_mobile::time();
    int max = 1;

    //    for (int i = 0; i < max; ++i) {
    //      auto time5 = paddle_mobile::time();
    //      vec_result = paddle_mobile.Predict(input, dims);
    //      auto time6 = paddle_mobile::time();
    //      std::cout << "300 predict cost :第" << i << ": "
    //                << paddle_mobile::time_diff(time5, time6) << "ms" <<
    //                std::endl;
    //    }
    //    auto time4 = paddle_mobile::time();
    //
    //    std::cout << "300 predict cost :"
    //              << paddle_mobile::time_diff(time3, time4) / max << "ms"
    //              << std::endl;
    //    auto biggest =
    //        std::max_element(std::begin(vec_result), std::end(vec_result));
    //    std::cout << "300 Max element is " << *biggest << " at position "
    //              << std::distance(std::begin(vec_result), biggest) <<
    //              std::endl;
    //
    //    // 500*500
    //    std::vector<float> vec_result2;
    //
    //    std::vector<float> input2;
    //    std::vector<int64_t> dims2{1, 1, 500, 500};
    //    GetInput<float>(g_test_image_1x3x224x224, &input2, dims2);
    //
    //    time3 = paddle_mobile::time();
    //    for (int i = 0; i < max; ++i) {
    //      auto time5 = paddle_mobile::time();
    //      vec_result2 = paddle_mobile.Predict(input2, dims2);
    //      auto time6 = paddle_mobile::time();
    //      std::cout << "500 predict cost :第" << i << ": "
    //                << paddle_mobile::time_diff(time5, time6) << "ms" <<
    //                std::endl;
    //    }
    //
    //    time4 = paddle_mobile::time();
    //    std::cout << "500 predict cost :"
    //              << paddle_mobile::time_diff(time3, time4) / max << "ms"
    //              << std::endl;
    //    biggest = std::max_element(std::begin(vec_result2),
    //    std::end(vec_result2)); std::cout << "500 Max element is " << *biggest
    //    << " at position "
    //              << std::distance(std::begin(vec_result2), biggest) <<
    //              std::endl;
    //
    //    // 1000*1000
    //
    //    std::vector<float> vec_result3;
    //    std::vector<float> input3;
    //    std::vector<int64_t> dims3{1, 1, 1000, 1000};
    //    GetInput<float>(g_test_image_1x3x224x224, &input3, dims3);
    //
    //    time3 = paddle_mobile::time();
    //
    //    for (int i = 0; i < max; ++i) {
    //      auto time5 = paddle_mobile::time();
    //      vec_result3 = paddle_mobile.Predict(input3, dims3);
    //      auto time6 = paddle_mobile::time();
    //      std::cout << "1000*1000 predict cost :第" << i << ": "
    //                << paddle_mobile::time_diff(time5, time6) << "ms" <<
    //                std::endl;
    //    }
    //    time4 = paddle_mobile::time();
    //    std::cout << "1000*1000 predict cost :"
    //              << paddle_mobile::time_diff(time3, time4) / max << "ms"
    //              << std::endl;
    //    biggest = std::max_element(std::begin(vec_result3),
    //    std::end(vec_result3)); std::cout << "1000*1000 Max element is " <<
    //    *biggest << " at position "
    //              << std::distance(std::begin(vec_result3), biggest) <<
    //              std::endl;

    // 224*224
    std::vector<float> vec_result4;
    std::vector<float> input4;
    std::vector<int64_t> dims4{1, 1, 300, 300};
    GetInput<float>(g_test_image_1x3x224x224, &input4, dims4);

    time3 = paddle_mobile::time();
    for (int i = 0; i < max; ++i) {
      auto time5 = paddle_mobile::time();
      vec_result4 = paddle_mobile.Predict(input4, dims4);
      auto time6 = paddle_mobile::time();
      std::cout << "300*300 predict cost :第" << i << ": "
                << paddle_mobile::time_diff(time5, time6) << "ms" << std::endl;
    }

    auto time4 = paddle_mobile::time();
    std::cout << "300*300 predict cost :"
              << paddle_mobile::time_diff(time3, time4) / max << "ms"
              << std::endl;
    //    biggest = std::max_element(std::begin(vec_result4),
    //    std::end(vec_result4)); std::cout << "224*224 Max element is " <<
    //    *biggest << " at position "
    //              << std::distance(std::begin(vec_result4), biggest) <<
    //              std::endl;
  }

  return 0;
}
