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
#include "operators/depthwise_conv_op.h"

int main() {
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  //  ../models/image_classification_resnet.inference.model
  auto program = loader.Load(g_mobilenet_ssd);

  PADDLE_MOBILE_ENFORCE(program.originProgram != nullptr,
                        "program file read fail");

  Executor4Test<paddle_mobile::CPU, paddle_mobile::operators::DepthwiseConvOp<
                                        paddle_mobile::CPU, float>>
      executor(program, "depthwise_conv2d");

  paddle_mobile::framework::LoDTensor input;
  // GetInput<float>(g_test_image_1x3x224x224, &input, {1, 3, 224, 224});
  // use SetupTensor if not has local input image .
  SetupTensor<float>(&input, {1, 32, 150, 150}, static_cast<float>(0),
                     static_cast<float>(1));
  auto input_ptr = input.data<float>();
  auto out_ddim = paddle_mobile::framework::make_ddim({1, 32, 150, 150});
  auto output = executor.Predict(input, "batch_norm_0.tmp_3",
                                 "depthwise_conv2d_0.tmp_0", out_ddim);

  auto output_ptr = output->data<float>();
  for (int j = 0; j < output->numel(); ++j) {
    DLOG << " value of output: " << output_ptr[j];
  }
  return 0;
}
