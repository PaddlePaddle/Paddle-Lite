// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/host/expand_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T, PrecisionType PType>
void ExpandCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::ExpandParam>();
  const auto* x = param.X;
  auto* out = param.Out;
  std::vector<int> expand_times = param.expand_times;

  const T* src = x->template data<T>();
  T* dst = out->template mutable_data<T>();

  int dims = expand_times.size();
  DDim in_shape = x->dims();

  int inner_num = 1;
  int i = dims - 1;
  int outer_num = in_shape.count(0, i);
  inner_num *= in_shape[i];
  for (int j = 0; j < outer_num; ++j) {
    for (int k = 0; k < expand_times[i]; ++k) {
      memcpy(dst + (j * expand_times[i] + k) * inner_num,
             src + j * inner_num,
             sizeof(T) * inner_num);
    }
  }
  inner_num *= expand_times[i];
  for (int i = dims - 2; i >= 0; --i) {
    int outer_num = in_shape.count(0, i);
    inner_num *= in_shape[i];
    for (int j = outer_num - 1; j >= 0; --j) {
      for (int k = expand_times[i] - 1; k >= 0; --k) {
        memcpy(dst + (j * expand_times[i] + k) * inner_num,
               dst + j * inner_num,
               sizeof(T) * inner_num);
      }
    }
    inner_num *= expand_times[i];
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using expand_float =
    paddle::lite::kernels::host::ExpandCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(expand, kHost, kFloat, kAny, expand_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();
