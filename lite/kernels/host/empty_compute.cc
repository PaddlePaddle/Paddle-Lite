// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/empty_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void EmptyCompute::Run() {
  auto& param = *param_.get_mutable<param_t>();
  auto output = param.Out;
  auto output_dims = output->dims();
  if (param.dtype == static_cast<int32_t>(lite::core::FluidType::BOOL)) {
    output->set_precision(PRECISION(kBool));
    output->template mutable_data<bool>();
  } else if (param.dtype == static_cast<int32_t>(lite::core::FluidType::FP32)) {
    output->set_precision(PRECISION(kFloat));
    output->template mutable_data<float>();
  } else if (param.dtype ==
             static_cast<int32_t>(lite::core::FluidType::INT32)) {
    output->set_precision(PRECISION(kInt32));
    output->template mutable_data<int32_t>();
  } else if (param.dtype ==
             static_cast<int32_t>(lite::core::FluidType::INT64)) {
    output->set_precision(PRECISION(kInt64));
    output->template mutable_data<int64_t>();
  } else {
    output->set_precision(PRECISION(kInt32));
    output->template mutable_data<int32_t>();
  }

  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    empty, kHost, kAny, kNCHW, paddle::lite::kernels::host::EmptyCompute, def)
    .BindInput("ShapeTensor",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindInput("ShapeTensorList",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .Finalize();
