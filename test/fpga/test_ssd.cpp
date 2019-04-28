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
#include <iomanip>
#include <iostream>
#include <typeindex>
#include <typeinfo>
#include "../test_include.h"

#include "fpga/KD/float16.hpp"
#include "fpga/KD/llapi/zynqmp_api.h"

static const char* g_ssd = "../models/ssd";

int main() {
  zynqmp::open_device();

  paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;
  std::string dir = std::string(g_ssd);
  std::string model = std::string(g_ssd) + "/model";
  std::string params = std::string(g_ssd) + "/params";

  // if (paddle_mobile.Load(dir, true)) {
  if (paddle_mobile.Load(model, params, true)) {
    Tensor input_tensor;
    SetupTensor<float>(&input_tensor, {1, 3, 299, 299}, static_cast<float>(1),
                       static_cast<float>(1));
    float* data = input_tensor.mutable_data<float>({1, 3, 299, 299});

    paddle_mobile.Predict(input_tensor);
    auto result_ptr = paddle_mobile.Fetch();
    float* result_data = result_ptr->data<float>();
  }
  return 0;
}
