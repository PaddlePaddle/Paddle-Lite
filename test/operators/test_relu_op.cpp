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

#include "../executor_for_test.h"
#include "../test_include.h"

int main() {
  paddle_mobile::Loader<paddle_mobile::CPU> loader;
  //  ../models/image_classification_resnet.inference.model
  auto program = loader.Load(g_mobilenet_ssd);

  PADDLE_MOBILE_ENFORCE(program.originProgram != nullptr,
                        "program file read fail");

  Executor4Test<paddle_mobile::CPU,
                paddle_mobile::operators::ReluOp<paddle_mobile::CPU, float>>
      executor(program, "relu");

  paddle_mobile::framework::Tensor input;
  SetupTensor<float>(&input, {1, 2, 3, 4}, static_cast<float>(-1),
                     static_cast<float>(1));

  auto out_ddim = paddle_mobile::framework::make_ddim({1, 2, 3, 4});
  auto output = executor.predict(input, "batch_norm_0.tmp_2",
                                 "batch_norm_0.tmp_3", out_ddim);

  auto output_ptr = output->data<float>();
  for (int j = 0; j < output->numel(); ++j) {
    DLOG << " value of output: " << output_ptr[j];
  }
  return 0;
}
