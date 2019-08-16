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

#ifdef ELEMENTWISEADD_OP

#pragma once

#include "operators/math/element_wise.h"
#include "operators/op_param.h"
#if defined(__ARM_NEON__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace paddle_mobile {
namespace operators {

template <typename T>
inline void ElementwiseAddCompute(const ElementwiseAddParam<CPU> &param) {
  const framework::Tensor *input_x = param.InputX();
  const framework::Tensor *input_y = param.InputY();
  framework::Tensor *output = param.Out();
  int axis = param.Axis();
  math::AddElememtWise<IDENTITY>(input_x, input_y, axis, output);
}

template <typename Dtype, ActivationType Act>
struct AddElememtWiseStruct {
  void operator()(const Tensor *X, const Tensor *Y, const int Axis,
                  Tensor *Out) {}
};

template <ActivationType Act>
struct AddElememtWiseStruct<int, Act> {
  void operator()(const Tensor *input, const Tensor *bias, const int Axis,
                  Tensor *output) {
    const auto &x_dims = input->dims();
    const auto &y_dims = bias->dims();
    const int *input_data = input->data<int>();
    const int *bias_data = bias->data<int>();
    int *output_data = output->mutable_data<int>();

    if (x_dims == y_dims) {
      size_t channels = 1;
      size_t elementwise_num = 1;
      for (int i = 0; i < y_dims.size(); ++i) {
        channels *= y_dims[i];
      }
#pragma omp parallel for
      for (int j = 0; j < channels; ++j) {
        size_t offset = (0 * channels + j) * elementwise_num;
        const int *input = input_data + offset;
        const int bias = bias_data[j];
        int *output = output_data + offset;
        for (int k = 0; k < elementwise_num; ++k) {
          output[k] = math::Active<Act>(input[k] + bias);
        }
      }
    }
  }
};

template class ElementwiseAddKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
