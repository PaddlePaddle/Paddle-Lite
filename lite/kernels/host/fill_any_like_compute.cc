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

#include "lite/kernels/host/fill_constant_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void FillAnyLikeCompute::FillAnyData() {
  auto& param = *param_.get_mutable<param_t>();
  T value = param.value;
  auto data = param.out->template mutable_data<T>();
  for (int i = 0; i < param.out->numel(); i++) {
    data[i] = value;
  }
}

void FillAnyLikeCompute::Run() {
  auto& param = *param_.get_mutable<param_t>();
  if (param.dtype == static_cast<int32_t>(lite::core::FluidType::FP32)) {
    FillAnyData<float>();
  } else if (param.dtype ==
             static_cast<int32_t>(lite::core::FluidType::INT32)) {
    FillAnyData<int32_t>();
  } else if (param.dtype == static_cast<int32_t>(lite::core::FluidType::INT8)) {
    FillAnyData<int8_t>();
  } else if (param.dtype ==
             static_cast<int32_t>(lite::core::FluidType::INT64)) {
    FillAnyData<int64_t>();
  } else {
    LOG(FATAL) << "not supported dtype " << param.dtype;
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fill_any_like_compute,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::FillAnyLikeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindPaddleOpVersion("fill_any_like_compute", 1)
    .Finalize();

REGISTER_LITE_KERNEL(fill_zeros_like_compute,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::FillAnyLikeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindPaddleOpVersion("fill_any_like_compute", 1)
    .Finalize();
