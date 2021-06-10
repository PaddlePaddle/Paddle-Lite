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

#include "lite/kernels/host/expand_as_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T, PrecisionType PType>
void ExpandAsCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::ExpandAsParam>();
  const auto* x = param.X;
  auto* out = param.Out;
  const auto* target = param.Target;
  std::vector<int> expand_times;
  const T* src = x->template data<T>();
  T* dst = out->template mutable_data<T>();

  for (int i = 0; i < target->dims().size(); ++i) {
    int times = target->dims()[i] / x->dims()[i];
    expand_times.push_back(times);
  }
  int dims = target->dims().size();
  DDim in_shape = x->dims();

  int inner_num = 1;
  int pos = dims - 1;
  int outer_num = in_shape.count(0, pos);
  inner_num *= in_shape[pos];
  for (int j = 0; j < outer_num; ++j) {
    for (int k = 0; k < expand_times[pos]; ++k) {
      memcpy(dst + (j * expand_times[pos] + k) * inner_num,
             src + j * inner_num,
             sizeof(T) * inner_num);
    }
  }
  inner_num *= expand_times[pos];
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
using expand_as_float =
    paddle::lite::kernels::host::ExpandAsCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(expand_as, kHost, kFloat, kAny, expand_as_float, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kAny))})
    .BindInput("target_tensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kAny))})
    .Finalize();

using expand_as_int64 =
    paddle::lite::kernels::host::ExpandAsCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(expand_as, kHost, kFloat, kAny, expand_as_int64, int64)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kInt64),
                                      DATALAYOUT(kAny))})
    .BindInput("target_tensor",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kInt64),
                                       DATALAYOUT(kAny))})
    .Finalize();
