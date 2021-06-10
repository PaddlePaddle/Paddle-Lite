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

#include "lite/kernels/host/where_compute.h"
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void where_kernel(const operators::WhereParam& param) {
  auto* x = param.x;
  auto* y = param.y;
  auto* condition = param.condition;
  auto* out = param.out;
  auto dims = x->dims();
  auto numel = dims.production();
  const T* x_data = x->template data<T>();
  const T* y_data = y->template data<T>();
  const bool* cond_data = condition->template data<bool>();
  T* out_data = out->template mutable_data<T>();
  for (int i = 0; i < numel; i++) {
    out_data[i] = cond_data[i] ? x_data[i] : y_data[i];
  }
}

void WhereCompute::Run() {
  auto& param = this->Param<operators::WhereParam>();
  switch (param.x->precision()) {
    case PRECISION(kFloat):
      where_kernel<float>(param);
      break;
    case PRECISION(kInt32):
      where_kernel<int32_t>(param);
      break;
    case PRECISION(kInt64):
      where_kernel<int64_t>(param);
      break;
    case PRECISION(kInt8):
      where_kernel<int8_t>(param);
      break;
    case PRECISION(kBool):
      where_kernel<bool>(param);
      break;
    default:
      LOG(FATAL) << "Where does not implement for the "
                 << "input type:" << static_cast<int>(param.x->precision());
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    where, kHost, kAny, kAny, paddle::lite::kernels::host::WhereCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindInput("Condition",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kBool),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();
