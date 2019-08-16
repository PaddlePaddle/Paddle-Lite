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
      paddle_mobile.Load(std::string(g_vgg16_ssd_combined) + "/model",
                         std::string(g_vgg16_ssd_combined) + "/params", false);
  if (isok) {
    auto time2 = paddle_mobile::time();
    std::cout << "load cost :" << paddle_mobile::time_diff(time1, time1) << "ms"
              << std::endl;

    std::vector<int64_t> dims{1, 3, 300, 300};
    Tensor input_tensor;
    SetupTensor<float>(&input_tensor, {1, 3, 300, 300}, static_cast<float>(0),
                       static_cast<float>(1));

    std::vector<float> input(input_tensor.data<float>(),
                             input_tensor.data<float>() + input_tensor.numel());

    auto vec_result = paddle_mobile.Predict(input, dims);

    DLOG << vec_result;
  }

  return 0;
}
