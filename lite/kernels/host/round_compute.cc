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

#include "lite/kernels/host/round_compute.h"
#include "lite/kernels/host/elementwise_op_func.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
void RoundCompute<T>::Run() {
  auto& param = Param<operators::RoundParam>();
  const lite::Tensor* input = param.X;
  lite::Tensor* output = param.Out;

  output->Resize(input->dims());

  auto out_num = output->dims().production();
  for (int i = 0; i < out_num; ++i) {
    output->mutable_data<T>()[i] = std::round(input->data<T>()[i]);
  }

  return;
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(round,
                     kHost,
                     kAny,
                     kNCHW,
                     paddle::lite::kernels::host::RoundCompute<float>,
                     fp32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindPaddleOpVersion("round", 1)
    .Finalize();
