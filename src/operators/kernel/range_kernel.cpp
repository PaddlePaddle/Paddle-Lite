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

#ifdef RANGE_OP

#include "operators/kernel/range_kernel.h"
#include "framework/data_type.h"

namespace paddle_mobile {
namespace operators {

template <>
bool RangeKernel<CPU, float>::Init(RangeParam<CPU>* param) {
  return true;
}

template <>
void RangeKernel<CPU, float>::Compute(const RangeParam<CPU>& param) {
  int start = param.Start()->data<int>()[0];
  int end = param.End()->data<int>()[0];
  int step = param.Step()->data<int>()[0];
  auto* out = param.Output();

  int64_t size = 0;
  GetSize(start, end, step, &size);
  out->Resize(framework::make_ddim({size}));
  auto* out_data = out->mutable_data<int>();
  auto value = start;
  for (int64_t i = 0; i < size; ++i) {
    out_data[i] = value;
    value += step;
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif  // RANGE_OP
