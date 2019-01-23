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
#ifdef PADDLE_MOBILE_CL
#include <CL/cl.h>
#include <cmath>
#include <iostream>
#include "../../test_helper.h"
#include "../../test_include.h"
#include "framework/cl/cl_tensor.h"
#include "operators/activation_op.h"

namespace paddle_mobile {

void Sigmoid(const framework::Tensor *X, framework::Tensor *Y) {
  const float *x = X->data<float>();
  float *y = Y->mutable_data<float>();

  for (int i = 0; i < X->numel(); ++i) {
    y[i] = 1.f / (1.f + exp(-x[i]));
  }
}

int TestSigmoidOp(const std::vector<int> input_shape) {
  paddle_mobile::PaddleMobile<paddle_mobile::GPU_CL> paddle_mobile;
  //    paddle_mobile.SetThreadNum(4);
  auto time1 = paddle_mobile::time();
#ifdef PADDLE_MOBILE_CL
  paddle_mobile.SetCLPath("/data/local/tmp/bin");
#endif
  auto isok = paddle_mobile.Load(std::string(g_sigmoid), true, false, 1, true);
  if (!isok) {
    exit(1);
  }
  framework::DDim dims = framework::make_ddim(input_shape);
  Tensor input_tensor;
  SetupTensor<float>(&input_tensor, framework::make_ddim(input_shape),
                     static_cast<float>(0), static_cast<float>(1));
  Tensor input_tensor2 = input_tensor;

  //  paddle_mobile.Feed(input_tensor, "feed");
  paddle_mobile.Predict(input_tensor);
  auto output = paddle_mobile.Fetch();

  framework::Tensor output_cmp;
  float *output_cmp_data = output_cmp.mutable_data<float>(output->dims());

  Sigmoid(&input_tensor2, &output_cmp);

  const float *output_data = output->data<float>();
  for (int i = 0; i < output->numel(); ++i) {
    float gap = output_data[i] - output_cmp_data[i];
    if (std::abs(gap / (output_data[i] + 1e-5)) > 1e-2) {
      LOG(kLOG_INFO) << "input_tensor[" << i
                     << "]=  " << input_tensor.data<float>()[i] << "   "
                     << input_tensor2.data<float>()[i] << "output_data[" << i
                     << "] = " << output_data[i] << ", output_cmp_data[" << i
                     << "] = " << output_cmp_data[i];
      exit(1);
    }
  }
  return 0;
}

}  // namespace paddle_mobile
#endif
int main() {
#ifdef PADDLE_MOBILE_CL
  paddle_mobile::TestSigmoidOp({1, 1, 1000, 1000});
  paddle_mobile::TestSigmoidOp({1, 3, 11, 22});
  paddle_mobile::TestSigmoidOp({1, 32, 112, 112});
  std::cout << "test sigmoid op cl pass." << std::endl;
#endif
  return 0;
}
