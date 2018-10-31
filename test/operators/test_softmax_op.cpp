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

#include "operators/softmax_op.h"

int main() {
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  auto program = loader.Load(std::string(g_mobilenet));
  if (program.originProgram == nullptr) {
    DLOG << "program read file";
  }
  Executor4Test<paddle_mobile::CPU,
                paddle_mobile::operators::SoftmaxOp<paddle_mobile::CPU, float>>
      executor(program, "softmax");
  paddle_mobile::framework::Tensor input;
  SetupTensor<float>(&input, {1, 1000}, static_cast<float>(0),
                     static_cast<float>(1));
  auto out_ddim = paddle_mobile::framework::make_ddim({1, 1000});
  auto output =
      executor.Predict(input, "reshape_0.tmp_0", "softmax_0.tmp_0", out_ddim);
  auto *output_ptr = output->data<float>();
  for (int j = 0; j < output->numel(); ++j) {
    DLOG << " value of output: " << output_ptr[j];
  }

  return 0;
}
