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

#ifdef PRELU_OP

#include "operators/kernel/prelu_kernel.h"
#include <operators/math/transform.h>

namespace paddle_mobile {
namespace operators {

template <typename T>
struct PReluFunctor {
  explicit PReluFunctor(float slope) { this->slope_ = slope; }
  inline T operator()(T in) const { return in > 0 ? in : in * slope_; }

  float slope_ = 0.0f;
};

/*
 * @b 特化到具体平台的实现, param 从 op 层传入
 * */
template <>
void PReluKernel<CPU, float>::Compute(const PReluParam &param) const {
  auto *x = param.InputX();
  auto *alpha = param.InputAlpha();
  auto *out = param.Out();
  std::string mode = param.Mode();
  const auto *x_ptr = x->data<float>();
  auto *o_ptr = out->mutable_data<float>();
  const auto *alpha_ptr = alpha->data<float>();
  int numel = x->numel();
  auto dim = x->dims();
  int index = 0;
  int i = 0;
  int temp = 0;
  if (mode == "channel") {
    temp = numel / (dim[0] * dim[1]);
    #pragma omp parallel for
    for (i = 0; i < numel; i++) {
      index = (i / temp) % dim[1];
      o_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha_ptr[index] * x_ptr[i];
    }
  } else if (mode == "element") {
    #pragma omp parallel for
    for (i = 0; i < numel; i++) {
      o_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha_ptr[i] * x_ptr[i];
    }
  } else {
    #pragma omp parallel for
    for (i = 0; i < numel; i++) {
      o_ptr[i] = x_ptr[i] > 0 ? x_ptr[i] : alpha_ptr[0] * x_ptr[i];
    }
  }
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
