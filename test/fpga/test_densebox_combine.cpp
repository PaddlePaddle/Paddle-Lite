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

#ifdef PADDLE_MOBILE_FPGA_V1
#include "fpga/V1/api.h"
#endif
#ifdef PADDLE_MOBILE_FPGA_V2
#include "fpga/V2/api.h"
#endif

static const char *g_densebox_combine = "../models/densebox";
int main() {
  paddle_mobile::fpga::open_device();
  paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;
  // paddle_mobile.SetThreadNum(4);
  if (paddle_mobile.Load(std::string(g_densebox_combine) + "/model",
                         std::string(g_densebox_combine) + "/params", true)) {
    // std::vector<float> input;
    // std::vector<int64_t> dims{1, 3, 512, 1024};
    // GetInput<float>(g_test_image_1x3x224x224_banana, &input, dims);

    // auto vec_result = paddle_mobile.Predict(input, dims);

    Tensor input_tensor;
    SetupTensor<float>(&input_tensor, {1, 3, 512, 1024}, static_cast<float>(0),
                       static_cast<float>(1));
    // readStream(g_image_src_float,
    //           input_tensor.mutable_data<float>({1, 3, 224, 224}));
    paddle_mobile.FeedData(input_tensor);
    paddle_mobile.Predict_To(-1);
  }

  return 0;
}
