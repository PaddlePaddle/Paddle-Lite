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
#include <sstream>
#include "../test_helper.h"
#include "../test_include.h"
#include "io/paddle_mobile_wrap.h"

int main(int argc, char *argv[]) {
#if defined(PADDLE_MOBILE_FPGA)
  paddle_mobile::wrap::Net<paddle_mobile::wrap::CPU> *net =
      new paddle_mobile::wrap::Net<paddle_mobile::wrap::CPU>();
  net->Load("./checked_model/model", "./checked_model/params", false, false, 1,
            true);
  int size = 1 * 3 * 64 * 64;
  float *data = new float[size];
  for (int i = 0; i < size; i++) {
    data[i] = 0.0;
  }
  std::vector<int64_t> shape{1, 3, 64, 64};
  paddle_mobile::wrap::Tensor input(data,
                                    paddle_mobile::wrap::make_ddim(shape));
  net->Feed("data", input);
  net->Predict();
  auto output = net->Fetch("save_infer_model/scale_0");
  int output_size = 1;
  std::cout << "output shape: ";
  for (int i = 0; i < output->dims().size(); i++) {
    std::cout << output->dims()[i] << " ";
    output_size *= output->dims()[i];
  }
  std::cout << std::endl;
  std::cout << "output data: ";
  for (int i = 0; i < output_size; i++) {
    std::cout << output->data<float>()[i] << std::endl;
  }
#endif
}
