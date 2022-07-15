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

#include "lite/kernels/host/reverse_compute.h"
#include <string>
#include <vector>
#include "lite/backends/host/math/reverse.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void ReverseCompute<T>::Run() {
  auto& param = Param<operators::ReverseParam>();
  auto input = param.X;
  auto output = param.Out;
  for (auto& axis : param.Axis)
    if (axis < 0) axis += input->dims().size();
  lite::host::math::reverse_func<T>(input, param.Axis, output);
#ifdef LITE_WITH_PROFILE
  kernel_func_name_ = "reverse_func";
#endif
  return;
}

void ReverseArrayCompute::Run() {
  auto& param = Param<operators::ReverseParam>();
  auto x_array = param.X_array;
  auto out_array = param.Out_array;
  out_array->resize(x_array->size());
  for (size_t i = 0; i < x_array->size(); i++) {
    out_array->at(x_array->size() - i - 1).CopyDataFrom(x_array->at(i));
  }
#ifdef LITE_WITH_PROFILE
  kernel_func_name_ = "reverse_func";
#endif
  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(reverse,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::ReverseCompute<float>,
                     fp32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindPaddleOpVersion("reverse", 1)
    .Finalize();

REGISTER_LITE_KERNEL(reverse,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::ReverseArrayCompute,
                     def_tensor_array)
    .BindInput("X",
               {LiteType::GetTensorListTy(TARGET(kHost),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorListTy(TARGET(kHost),
                                           PRECISION(kAny),
                                           DATALAYOUT(kAny))})
    .BindPaddleOpVersion("reverse", 1)
    .Finalize();
