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

void feed(PaddleMobile<paddle_mobile::GPU_CL> *paddle_mobile, const DDim &dims,
          std::string image_path, std::string feed_name) {
  float *input_data_array = new float[product(dims)];
  std::ifstream in(image_path, std::ios::in);
  for (int i = 0; i < product(dims); i++) {
    float num;
    in >> num;
    input_data_array[i] = num;
  }
  in.close();
  framework::Tensor input_tensor(input_data_array, dims);
  DLOG << feed_name << " : " << input_tensor;
  paddle_mobile->Feed(feed_name, input_tensor);
}

int main() {
  paddle_mobile::PaddleMobile<paddle_mobile::GPU_CL> paddle_mobile;
  auto time1 = paddle_mobile::time();
#ifdef PADDLE_MOBILE_CL
  paddle_mobile.SetCLPath("/data/local/tmp/bin");
#endif

  if (paddle_mobile.Load(std::string("../models/nanbiannv") + "/model",
                         std::string("../models/nanbiannv") + "/params",
                         true)) {
    auto time2 = paddle_mobile::time();
    std::cout << "load cost :" << paddle_mobile::time_diff(time1, time2) << "ms"
              << std::endl;

    std::vector<float> input;
    feed(&paddle_mobile, {1, 3, 256, 256}, "../images/input_1_3_256_256",
         "image");

    auto time3 = paddle_mobile::time();
    paddle_mobile.Predict();
    auto time4 = paddle_mobile::time();

    std::cout << "predict cost :" << paddle_mobile::time_diff(time3, time4)
              << "ms" << std::endl;
  }

  auto rgb = paddle_mobile.Fetch("rgb");
  auto mask = paddle_mobile.Fetch("mask");
  LOG(kLOG_INFO) << "rgb" << *rgb;
  LOG(kLOG_INFO) << "mask" << *mask;
  return 0;
}
