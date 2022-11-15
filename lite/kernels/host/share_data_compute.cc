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

#include "lite/kernels/host/share_data_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

// template <typename T>
void ShareDataCompute::Run() {
  auto& param = Param<operators::ShareDataParam>();
  const lite::Tensor* input = param.X;
  lite::Tensor* output = param.Out;
  output->ShareDataWith(*input);
  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(share_data,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::ShareDataCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kAny))})
    .BindPaddleOpVersion("share_data", 1)
    .Finalize();
