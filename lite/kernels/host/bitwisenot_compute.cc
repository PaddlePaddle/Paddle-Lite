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

#include "lite/kernels/host/bitwisenot_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

void BitwiseNotCompute::Run() {
  auto& param = Param<param_t>();
  CHECK(param.X);
  const auto* input_data = param.X->data<bool>();
  auto* output_data = param.Out->mutable_data<bool>();
  for (int i = 0; i < param.X->numel(); ++i) {
    output_data[i] = ~input_data[i];
  }

  return;
}
}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(bitwise_not,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::BitwiseNotCompute,
                     bit_bl)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kBool))})
    .Finalize();
