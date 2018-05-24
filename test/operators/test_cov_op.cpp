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

#include <io>
#include "framework/executor_for_test.h"
#include "framework/test_helper.h"

int main() {
  paddle_mobile::Loader<paddle_mobile::CPU> loader;
  //  ../models/image_classification_resnet.inference.model
  auto program = loader.Load(std::string("../models/googlenet"));
  if (program.originProgram == nullptr) {
    DLOG << "program file read fail";
  }

  Executor4Test<paddle_mobile::CPU,
                paddle_mobile::operators::ConvOp<paddle_mobile::CPU, float>>
      executor(program, "conv2d");

  paddle_mobile::framework::Tensor input;
  SetupTensor<float>(&input, {1, 3, 32, 32}, static_cast<float>(0),
                     static_cast<float>(1));
  auto out_ddim = paddle_mobile::framework::make_ddim({1, 64, 56, 56});
  auto output = executor.predict(input, "data", "conv2d_0.tmp_0", out_ddim);

  auto output_ptr = output->data<float>();
  for (int j = 0; j < output->numel(); ++j) {
    DLOG << " value of output: " << output_ptr[j];
  }
  return 0;
}
