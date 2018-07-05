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
  const auto *input_x = param.InputX();
  auto *input_x_ptr = input_x->data<float>();
  auto *out = param.Out();
  auto *out_ptr = out->mutable_data<float>();

  if (param.Slopes().size() == 1) {
    PReluFunctor<float> func_(param.Slopes()[0]);
    math::Transform trans;
    trans(input_x_ptr, input_x_ptr + input_x->numel(), out_ptr, func_);
  } else if (param.Slopes().size() > 1) {
    const int dim_size = input_x->dims().size();
    switch (dim_size) {
      case 0:
        break;
      case 1: {
        const int input_width = input_x->dims()[0];
        math::Transform trans;

        #pragma omp parallel for
        for (int w = 0; w < input_width; ++w) {
          out_ptr[w] = input_x_ptr[w] * param.Slopes()[w];
        }
      } break;
      case 2: {
        const int input_height = input_x->dims()[0];
        const int input_width = input_x->dims()[1];

        math::Transform trans;
        #pragma omp parallel for
        for (int h = 0; h < input_height; ++h) {
          PReluFunctor<float> func_(param.Slopes()[h]);
          const float *ptr = input_x_ptr + h * input_width;
          float *optr = out_ptr + +h * input_width;
          trans(ptr, ptr + input_width, optr, func_);
        }
      } break;
      case 3: {
        const int chan_size = input_x->dims()[0];
        const int input_height = input_x->dims()[1];
        const int input_width = input_x->dims()[2];

        math::Transform trans;
        #pragma omp parallel for
        for (int c = 0; c < chan_size; ++c) {
          PReluFunctor<float> func_(param.Slopes()[c]);
          int size = input_height * input_width;
          const float *ptr = input_x_ptr + c * size;
          float *optr = out_ptr + c * size;
          trans(ptr, ptr + size, optr, func_);
        }
      } break;
      case 4:
      default: {
        const int batch_size = input_x->dims()[0];
        const int chan_size = input_x->dims()[1];
        const int input_height = input_x->dims()[2];
        const int input_width = input_x->dims()[3];
        math::Transform trans;

        #pragma omp parallel for
        for (int b = 0; b < batch_size; ++b) {
          for (int c = 0; c < chan_size; ++c) {
            PReluFunctor<float> func_(param.Slopes()[c]);
            int size = input_height * input_width;
            const float *ptr = input_x_ptr + b * c * size;
            float *optr = out_ptr + +b * c * size;
            trans(ptr, ptr + size, optr, func_);
          }
        }
      }  // case 3,default
      break;
    }
  }
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
