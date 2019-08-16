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
  //  ../../../test/models/googlenet
  //  ../../../test/models/mobilenet
  auto time1 = time();

  if (paddle_mobile.Load(std::string(g_yolo_vision) + "/model",
                         std::string(g_yolo_vision) + "/params", true)) {
    auto time2 = time();
    std::cout << "load cost :" << time_diff(time1, time1) << "ms" << std::endl;

    std::vector<int64_t> dims{1, 3, 416, 416};
    std::vector<float> input;

    GetInput<float>(g_test_image_1x3x416x416_vision_yolo_input, &input, dims);
    std::cout << "input.size():  " << input.size() << std::endl;
    for (int j = 0; j < 100; ++j) {
      std::cout << j << " :  " << input[j] << std::endl;
    }
    //        // 预热十次
    //        for (int i = 0; i < 10; ++i) {
    //            paddle_mobile.Predict(input, dims);
    //        }
    auto time3 = time();
    const vector<float> vector_out = paddle_mobile.Predict(input, dims);

    std::cout << "load cost :" << time_diff(time1, time1) << "ms" << std::endl;

    auto time4 = time();
    std::cout << "predict cost :" << time_diff(time3, time4) / 10 << "ms"
              << std::endl;
  }
  return 0;
}
