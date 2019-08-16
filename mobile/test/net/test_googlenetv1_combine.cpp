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
  if (paddle_mobile.Load(std::string(g_googlenetv1_combined) + "/model",
                         std::string(g_googlenetv1_combined) + "/params",
                         false)) {
    auto time2 = time();
    std::cout << "load cost :" << time_diff(time1, time1) << "ms" << std::endl;

    std::vector<float> input;
    std::vector<int64_t> dims{1, 3, 160, 160};
    GetInput<float>(g_img, &input, dims);

    for (int i = 0; i < input.size(); i += 1000) {
      std::cout << input[i] << std::endl;
    }
    //    auto vec_result = paddle_mobile.Predict(input, dims);
    //    std::vector<float>::iterator biggest =
    //        std::max_element(std::begin(vec_result), std::end(vec_result));
    //    std::cout << " Max element is " << *biggest << " at position "
    //              << std::distance(std::begin(vec_result), biggest) <<
    //              std::endl;

    //    // 预热十次
    //    for (int i = 0; i < 1; ++i) {
    //      auto vec_result = paddle_mobile.Predict(input, dims);
    //    }
    auto time3 = time();

    auto vec_result = paddle_mobile.Predict(input, dims);

    for (int j = 0; j < vec_result.size(); ++j) {
      std::cout << j << " : " << vec_result[j] << std::endl;
    }
    auto time4 = time();
    std::cout << "predict cost :" << time_diff(time3, time4) / 1 << "ms"
              << std::endl;
  }

  return 0;
}
