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
#include "../test_helper.h"
#include "../test_include.h"

int main() {
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  paddle_mobile.SetThreadNum(4);
  bool optimize = true;
  auto time1 = time();
  if (paddle_mobile.Load(g_googlenet, optimize)) {
    auto time2 = time();
    DLOG << "load cost: " << time_diff(time1, time1) << "ms";
    std::vector<float> input;
    std::vector<int64_t> dims{1, 3, 224, 224};
    GetInput<float>(g_test_image_1x3x224x224, &input, dims);
    auto time3 = time();
    auto vec_result = paddle_mobile.Predict(input, dims);
    auto time4 = time();

    DLOG << "predict cost :" << time_diff(time3, time4) << "ms";
  }
  return 0;
}
