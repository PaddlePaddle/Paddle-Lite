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

#include "lite/kernels/host/cos_compute.h"
#include <algorithm>
#include <cmath>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void CosCompute::Run() {
  auto& param = Param<operators::CosParam>();
  const float* x_data = param.X->data<float>();
  float* output_data = param.Out->mutable_data<float>();
  DDim x_dims = param.X->dims();
  for (int64_t i = 0; i < x_dims.production(); i++) {
    output_data[i] = std::cos(x_data[i]);
  }
#ifdef LITE_WITH_PROFILE
  kernel_func_name_ = "cos_func";
#endif
  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    cos, kHost, kFloat, kNCHW, paddle::lite::kernels::host::CosCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
