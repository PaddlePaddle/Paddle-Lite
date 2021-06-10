// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/fill_zeros_like_compute.h"
#include <cstring>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T>
void FillZerosLikeCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  auto* out = param.Out;
  auto* out_data = out->template mutable_data<T>();
  memset(out_data, 0, out->numel() * sizeof(T));
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(fill_zeros_like,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::FillZerosLikeCompute<float>,
                     float32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();

REGISTER_LITE_KERNEL(fill_zeros_like,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::FillZerosLikeCompute<int>,
                     int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

REGISTER_LITE_KERNEL(fill_zeros_like,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::FillZerosLikeCompute<int64_t>,
                     int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
