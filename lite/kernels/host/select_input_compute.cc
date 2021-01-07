// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/select_input_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void SelectInputCompute::Run() {
  auto& param = this->Param<param_t>();
  param.Out->ShareDataWith(*param.X[*param.Mask->data<int>()]);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(select_input,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::SelectInputCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindInput("Mask",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(select_input,
                     kHost,
                     kInt32,
                     kNCHW,
                     paddle::lite::kernels::host::SelectInputCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("Mask",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
