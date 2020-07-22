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

#include "lite/backends/cuda/cuda_utils.h"
#include "lite/backends/cuda/math/activation.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/sigmoid_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

template <typename T, PrecisionType Ptype>
void SigmoidCompute<T, Ptype>::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  int num = static_cast<int>(param.X->numel());
  auto input = param.X->template data<T>();
  auto output = param.Out->template mutable_data<T>(TARGET(kCUDA));

  lite::cuda::math::sigmoid<T>(num, input, output, stream);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using SigmoidFp32 =
    paddle::lite::kernels::cuda::SigmoidCompute<float, PRECISION(kFloat)>;

using SigmoidFp16 =
    paddle::lite::kernels::cuda::SigmoidCompute<half, PRECISION(kFP16)>;

REGISTER_LITE_KERNEL(sigmoid, kCUDA, kFloat, kNCHW, SigmoidFp32, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();

REGISTER_LITE_KERNEL(sigmoid, kCUDA, kFP16, kNCHW, SigmoidFp16, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA), PRECISION(kFP16))})
    .Finalize();
