/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/kernels/cuda/scale_compute.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

void ScaleCompute::Run() {
  auto& param = Param<operators::ScaleParam>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  const float* x_data = param.x->data<float>();
  float* output_data = param.output->mutable_data<float>(TARGET(kCUDA));
  DDim x_dims = param.x->dims();
  bool bias_after_scale = param.bias_after_scale;
  float scale = param.scale;
  float bias = param.bias;
  if (!bias_after_scale) {
    bias *= scale;
  }
  lite::cuda::math::scale(
      x_dims.production(), x_data, output_data, scale, bias, stream);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    scale, kCUDA, kFloat, kNCHW, paddle::lite::kernels::cuda::ScaleCompute, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kCUDA))})
    .Finalize();
