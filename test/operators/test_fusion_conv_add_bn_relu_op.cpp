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
#include "../test_include.h"
#include "operators/fusion_conv_add_bn_relu_op.h"

int main() {
  paddle_mobile::framework::Loader<paddle_mobile::CPU> loader;
  //  ../models/image_classification_resnet.inference.model
  auto program = loader.Load(g_mobilenet, true);

  PADDLE_MOBILE_ENFORCE(program.originProgram != nullptr,
                        "program file read fail");

  Executor4Test<paddle_mobile::CPU,
                paddle_mobile::operators::FusionConvAddBNReluOp<
                    paddle_mobile::CPU, float>>
      executor(program, "fusion_conv_add_bn_relu", true);

  std::cout << "executor 4 test: " << std::endl;

  paddle_mobile::framework::Tensor input;
  GetInput<float>(g_test_image_1x3x224x224_banana, &input, {1, 3, 224, 224});
  //  // use SetupTensor if not has local input image .
  //  SetupTensor<float>(&input, {1, 3, 224, 224}, static_cast<float>(0),
  //                     static_cast<float>(1));

  DLOG << " fuck: " << input;

  auto out_ddim = paddle_mobile::framework::make_ddim({1, 32, 112, 112});
  std::cout << "before predict: " << std::endl;
  auto output =
      executor.Predict(input, "data", "conv2_1_dw_bn.tmp_2", out_ddim);
  std::cout << "after predict " << std::endl;
  auto output_ptr = output->data<float>();

  int stride = output->numel() / 100;
  for (int i = 0; i < 100; i++) {
    DLOG << " index:" << i * stride << " value: " << output_ptr[i * stride];
  }

  //  for (int i = 0; i < 100; i++) {
  //    DLOG << " index:" << i << " value: "<< output_ptr[i];
  //  }

  //  for (int j = 0; j < output->numel(); ++j) {
  //    std::cout << " (index: " << j << " value: " << output_ptr[j] << ") ";
  //  }
  std::cout << std::endl;
  return 0;
}
