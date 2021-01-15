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

#include "lite/kernels/host/assign_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void AssignCompute::Run() {
  auto& param = Param<param_t>();
  if (param.X != nullptr) {
    param.Out->CopyDataFrom(*param.X);
  } else if (param.X_array != nullptr) {
    auto x_array = param.X_array;
    auto out_array = param.Out_array;
    out_array->resize(x_array->size());
    for (size_t i = 0; i < x_array->size(); i++) {
      out_array->at(i).CopyDataFrom(x_array->at(i));
    }
  } else {
    LOG(FATAL) << "x or x_array of assign must be set.";
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    assign, kHost, kAny, kAny, paddle::lite::kernels::host::AssignCompute, def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kHost),
                                      PRECISION(kAny),
                                      DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost),
                                       PRECISION(kAny),
                                       DATALAYOUT(kAny))})
    .Finalize();

REGISTER_LITE_KERNEL(assign,
                     kHost,
                     kAny,
                     kAny,
                     paddle::lite::kernels::host::AssignCompute,
                     def_tensor_array)
    .BindInput("X",
               {LiteType::GetTensorListTy(TARGET(kHost),
                                          PRECISION(kAny),
                                          DATALAYOUT(kAny))})
    .BindOutput("Out",
                {LiteType::GetTensorListTy(TARGET(kHost),
                                           PRECISION(kAny),
                                           DATALAYOUT(kAny))})
    .Finalize();
