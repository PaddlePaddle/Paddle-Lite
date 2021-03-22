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

#include "lite/kernels/host/topk_compute.h"
#include "lite/backends/host/math/topk.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void TopkCompute::Run() {
  auto& param = Param<operators::TopkParam>();
  const float* x_data = param.X->data<float>();
  float* out_val = param.Out->mutable_data<float>();
  auto out_ind = param.Indices->mutable_data<int64_t>();
  DDim x_dims = param.X->dims();
  int K = param.K;
  int dim_size = x_dims.size();
  int m = x_dims.production() / x_dims[dim_size - 1];
  int n = x_dims[dim_size - 1];
  lite::host::math::topk(x_data, out_val, out_ind, m, n, K);
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    top_k, kHost, kFloat, kNCHW, paddle::lite::kernels::host::TopkCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Indices",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();
