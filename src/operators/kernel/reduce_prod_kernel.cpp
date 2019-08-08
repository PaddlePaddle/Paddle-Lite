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

#ifdef REDUCE_PROD_OP

#include "operators/kernel/reduce_prod_kernel.h"
#include <operators/reduce_prod_op.h>
#include <array>
#include "framework/data_type.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ReduceProdKernel<CPU, float>::Init(ReduceProdParam<CPU>* param) {
  return true;
}

template <>
void ReduceProdKernel<CPU, float>::Compute(const ReduceProdParam<CPU>& param) {
  auto* input = param.Input();
  if (input->type() == type_id<int>().hash_code()) {
    bool reduce_all = param.isReduceAll();
    auto* output = param.Output();
    auto dim = param.getDim();
    auto* out_data = output->mutable_data<int>();
    const auto* input_x_data = input->data<int>();

    auto dims = param.getDim();
    bool keep_dim = param.isKeepDim();

    if (reduce_all) {
      size_t stride = 1;
      for (int j = dim[0]; j < input->dims().size(); ++j) {
        stride *= input->dims()[j];
      }
      auto numel = output->numel();
      for (int i = 0; i < numel; i++) {
        int64_t mul = 1;
        for (int j = 0; j < stride; ++j, ++input_x_data) {
          mul *= (*input_x_data);
        }
        out_data[i] = mul;
      }
    } else {
      // todo
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // REDUCE_PROD_OP
