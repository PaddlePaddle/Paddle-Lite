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

#include "lite/kernels/cuda/dropout_compute.h"
#include <string>
#include "lite/backends/cuda/math/scale.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

void DropoutCompute::Run() {
  auto& param = Param<operators::DropoutParam>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  const float* x_data = param.x->data<float>();
  float* out_data = param.output->mutable_data<float>(TARGET(kCUDA));
  int num = param.x->dims().production();
  const float prob_data = param.dropout_prob;
  float scale = 1.0f;
  if (param.dropout_implementation == "downgrade_in_infer") {
    scale = 1.0f - prob_data;
  }
  lite::cuda::math::scale(num, x_data, out_data, scale, 0.f, stream);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(dropout,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::DropoutCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindInput("Seed", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Mask", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
