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

  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile(config);
  //    paddle_mobile.SetThreadNum(4);
  auto time1 = paddle_mobile::time();
#ifdef PADDLE_MOBILE_CL
  paddle_mobile.SetCLPath("/data/local/tmp/bin");
#endif

  auto isok = paddle_mobile.Load(std::string(g_super) + "/model",
                                 std::string(g_super) + "/params", true, false,
                                 1, false);

  //  auto isok = paddle_mobile.Load(std::string(g_mobilenet_mul), true);
  if (isok) {
    auto time2 = paddle_mobile::time();
    std::cout << "load cost :" << paddle_mobile::time_diff(time1, time2) << "ms"
              << std::endl;

    std::vector<float> input;
    std::vector<int64_t> dims{1, 1, 300, 300};
    GetInput<float>(g_yolo_img, &input, dims);

    std::vector<float> vec_result;

    auto time3 = paddle_mobile::time();
    int max = 1;
    for (int i = 0; i < max; ++i) {
      vec_result = paddle_mobile.Predict(input, dims);
    }
    auto time4 = paddle_mobile::time();

    std::cout << "predict cost :"
              << paddle_mobile::time_diff(time3, time4) / max << "ms"
              << std::endl;
    std::vector<float>::iterator biggest =
        std::max_element(std::begin(vec_result), std::end(vec_result));
    std::cout << " Max element is " << *biggest << " at position "
              << std::distance(std::begin(vec_result), biggest) << std::endl;
  }

  std::cout << "如果结果Nan请查看: test/images/g_test_image_1x3x224x224_banana "
               "是否存在?"
            << std::endl;
  return 0;
}
