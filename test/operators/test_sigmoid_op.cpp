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

#include "../../src/operators/kernel/sigmoid_kernel.h"
#include "../test_helper.h"
#include "./io.h"

int main() {
  paddle_mobile::framework::Tensor input;
  paddle_mobile::framework::Tensor output;
  DLOG << 1;
  SetupTensor<float>(&input, {1, 4, 60, 60}, static_cast<float>(0),
                     static_cast<float>(1));
  DLOG << 2;

  auto out_ddim = paddle_mobile::framework::make_ddim({1, 4, 60, 60});
  output.Resize(out_ddim);
  DLOG << 3;
  paddle_mobile::operators::sigmoid(&input, &output);
  DLOG << 4;
  auto *output_ptr = output.data<float>();
  for (int j = 0; j < output.numel(); ++j) {
    DLOG << " value of output: " << output_ptr[j];
  }
  DLOG << 5;
  return 0;
}
