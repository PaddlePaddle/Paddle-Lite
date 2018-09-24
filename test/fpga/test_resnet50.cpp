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

#include "../test_include.h"
static const char *g_resnet_combine = "../models/resnet50";

int main() {
  DLOG << paddle_mobile::fpga::open_device();
  paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;
  if (paddle_mobile.Load(std::string(g_resnet_combine) + "/model",
                         std::string(g_resnet_combine) + "/params", true)) {
    std::vector<int64_t> dims{1, 3, 224, 224};
    Tensor input_tensor;
    SetupTensor<float>(&input_tensor, {1, 3, 224, 224}, static_cast<float>(0),
                       static_cast<float>(1));

    std::vector<float> input(input_tensor.data<float>(),
                             input_tensor.data<float>() + input_tensor.numel());

    paddle_mobile.FeedData(input_tensor);
    paddle_mobile.Predict_To(-1);
    //    paddle_mobile.Predict_From(73);
    //    paddle_mobile.Predict_From_To(72, 73);

    DLOG << "Computation done";
    return 0;
  }
}
