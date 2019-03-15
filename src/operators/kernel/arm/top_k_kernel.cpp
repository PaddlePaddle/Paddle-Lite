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

#ifdef TOP_K_OP

#include <algorithm>
#include <iostream>
#include <vector>
#include "framework/context.h"
#include "operators/kernel/kernels.h"

namespace paddle_mobile {
namespace operators {

template <>
bool TopKKernel<CPU, float>::Init(TopKParam<CPU> *param) {
  return true;
}

template <>
void TopKKernel<CPU, float>::Compute(const TopKParam<CPU> &param) {
  const Tensor *input = param.input_;
  Tensor *output = param.output_;
  Tensor *indices = param.indices_;
  const float *input_data = input->data<float>();
  float *output_data = output->mutable_data<float>();
  int64_t *indices_data = indices->mutable_data<int64_t>();

  framework::DDim input_dims = input->dims();
  const size_t row = framework::product(
      framework::slice_ddim(input_dims, 0, input_dims.size() - 1));
  const size_t col = input_dims[input_dims.size() - 1];

  #pragma omp parallel for
  // num_threads(framework::threads())
  for (size_t i = 0; i < row; i++) {
    std::vector<std::pair<float, size_t>> vec(col);
    const float *input_ptr = input_data + i * col;
    float *output_ptr = output_data + i * param.k_;
    int64_t *indices_ptr = indices_data + i * param.k_;

    for (size_t j = 0; j < col; j++) {
      vec[j] = std::move(std::pair<float, size_t>(input_ptr[j], j));
    }
    std::partial_sort(
        vec.begin(), vec.begin() + param.k_, vec.end(),
        [](const std::pair<float, size_t> &l,
           const std::pair<float, size_t> &r) { return l.first > r.first; });
    for (int j = 0; j < param.k_; ++j) {
      output_ptr[j] = vec[j].first;
      indices_ptr[j] = static_cast<int64_t>(vec[j].second);
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // TOP_K_OP
