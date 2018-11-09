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
#include "operators/fusion_conv_add_relu_op.h"

int main() {
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  //  ../models/image_classification_resnet.inference.model
  auto program = loader.Load(g_googlenet, true);

  PADDLE_MOBILE_ENFORCE(program.originProgram != nullptr,
                        "program file read fail");

  Executor4Test<
      paddle_mobile::CPU,
      paddle_mobile::operators::FusionConvAddReluOp<paddle_mobile::CPU, float>>
      executor(program, "fusion_conv_add_relu", true);

  paddle_mobile::framework::Tensor input;
  GetInput<float>(g_test_image_1x3x224x224, &input, {1, 3, 224, 224});
  //  // use SetupTensor if not has local input image .
  //  SetupTensor<float>(&input, {1, 3, 224, 224}, static_cast<float>(0),
  //                     static_cast<float>(1));

  auto out_ddim = paddle_mobile::framework::make_ddim({1, 64, 112, 112});
  auto output = executor.Predict(input, "data", "conv2d_0.tmp_2", out_ddim);

  auto output_ptr = output->data<float>();
  for (int j = 0; j < 25; ++j) {
    DLOG << " value of output: " << output_ptr[j];
  }
  return 0;
}
