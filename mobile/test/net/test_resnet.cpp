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
#ifdef PADDLE_MOBILE_FPGA
  paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;
#endif

#ifdef PADDLE_MOBILE_CL
  paddle_mobile::PaddleMobile<paddle_mobile::GPU_CL> paddle_mobile;
  paddle_mobile.SetCLPath("/data/local/tmp/bin");
#else
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
#endif
  paddle_mobile.SetThreadNum(4);
  auto time1 = time();
  if (paddle_mobile.Load(g_resnet, true)) {
    auto time2 = time();
    std::cout << "load cost :" << time_diff(time1, time1) << "ms" << std::endl;
    std::vector<int64_t> dims{1, 3, 32, 32};
    Tensor input_tensor;
    SetupTensor<float>(&input_tensor, {1, 3, 32, 32}, static_cast<float>(0),
                       static_cast<float>(1));

    std::vector<float> input(input_tensor.data<float>(),
                             input_tensor.data<float>() + input_tensor.numel());
#ifndef PADDLE_MOBILE_FPGA
    //   预热十次
    //    for (int i = 0; i < 10; ++i) {
    //      paddle_mobile.Predict(input, dims);
    //    }
    auto time3 = time();
    //    for (int i = 0; i < 10; ++i) {
    paddle_mobile.Predict(input, dims);
    //    }
    auto time4 = time();
    std::cout << "predict cost :" << time_diff(time3, time4) << "ms"
              << std::endl;

#else
    auto time3 = time();
    paddle_mobile.FeedData(input_tensor);
    paddle_mobile.Predict_To(-1);
    /*paddle_mobile.Predict_From(10);
    auto tensor_ptr = paddle_mobile.FetchResult(9);
    std::cout << "Tensor element number for op[9]: " << tensor_ptr->numel()
              << std::endl;
    auto result_ptr = paddle_mobile.FetchResult();
    std::cout << "Result tensor element number: " << result_ptr->numel()
              << std::endl;

    auto time4 = time();
    std::cout << "predict cost :" << time_diff(time3, time4) << "ms"
              << std::endl;*/
#endif
  }
  return 0;
}
