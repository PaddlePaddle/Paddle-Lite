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

#include "../test_helper.h"
#include "../test_include.h"
#include "operators/transpose_op.h"
int main() {
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(std::string(g_mobilenet_ssd));
  if (program.originProgram == nullptr) {
    DLOG << "program read file";
  }
  Executor4Test<paddle_mobile::CPU, paddle_mobile::operators::TransposeOp<
                                        paddle_mobile::CPU, float>>
      executor(program, "transpose");
  paddle_mobile::framework::Tensor input;
  SetupTensor<float>(&input, {1, 2, 3, 4}, static_cast<float>(0),
                     static_cast<float>(1));
  auto input_ptr = input.data<float>();
  auto out_ddim = paddle_mobile::framework::make_ddim({1, 3, 4, 2});
  auto output =
      executor.Predict(input, "conv2d_22.tmp_1", "transpose_0.tmp_0", out_ddim);
  auto *output_ptr = output->data<float>();

  DLOG << "input : ";
  for (int j = 0; j < input.numel(); ++j) {
    DLOG << " index " << j << " : " << input_ptr[j];
  }

  DLOG << "output : ";
  for (int j = 0; j < output->numel(); ++j) {
    DLOG << " index " << j << " : " << output_ptr[j];
  }
  DLOG << " for example : ";
  DLOG << " you can check if input[16] == output[9] ";
  DLOG << " you can check if input[12] == output[1] ";
  return 0;
}
