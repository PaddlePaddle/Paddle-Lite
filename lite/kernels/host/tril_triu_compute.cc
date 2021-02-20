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

#include "lite/kernels/host/tril_triu_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <class T>
void TrilTriu(const T* in,
              const int64_t diagonal,
              const bool lower,
              const int64_t h,
              const int64_t w,
              T* out) {
  int64_t size = h * w;
  for (int64_t idx = 0; idx < size; idx++) {
    const int64_t row = idx / w;
    const int64_t col = idx % w;
    const bool mask = lower ? (col - row > diagonal) : (col - row < diagonal);
    out[idx] = mask ? 0 : in[idx];
  }
  return;
}

template <class T>
void TrilTriuCompute<T>::Run() {
  auto& param = this->template Param<param_t>();
  const lite::Tensor* x = param.x;
  lite::Tensor* out = param.out;
  int64_t diagonal = param.diagonal;
  bool lower = param.lower;

  const T* x_data = x->template data<T>();
  T* out_data = out->template mutable_data<T>();
  auto x_dims = x->dims();
  int64_t h = x_dims[x_dims.size() - 2];
  int64_t w = x_dims[x_dims.size() - 1];
  int64_t n = x_dims.production() / h / w;

  for (int64_t i = 0; i < n; i++) {
    TrilTriu(x_data, diagonal, lower, h, w, out_data);
    x_data += h * w;
    out_data += h * w;
  }
  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using TrilTriuFloat32 = paddle::lite::kernels::host::TrilTriuCompute<float>;
REGISTER_LITE_KERNEL(tril_triu, kHost, kAny, kNCHW, TrilTriuFloat32, float32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();
