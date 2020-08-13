/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "lite/kernels/arm/sequence_pool_grad_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void SequencePoolGradCompute::PrepareForRun() {}

void SequencePoolGradCompute::Run() {
  auto& param = Param<operators::SequencePoolGradParam>();
  auto& output_grad = param.Out_Grad;
  auto& x_grad = param.X_Grad;
  const auto* din_ptr = param.X->data<float>();
  const auto* dout_grad_ptr = output_grad->data<float>();
  const auto* index_grad_ptr = param.MaxIndex_Grad->data<int64_t>();
  float* x_grad_ptr = x_grad->mutable_data<float>();
  const auto pool_type = param.pool_type;
  const auto lod = param.X->lod()[0];
  int64_t width = param.X->numel() / param.X->dims()[0];
  if (pool_type == "SUM") {
    lite::arm::math::seq_pool_sum_grad(
        din_ptr, dout_grad_ptr, x_grad_ptr, lod, width);
  } else if (pool_type == "AVERAGE") {
    lite::arm::math::seq_pool_average_grad(
        din_ptr, dout_grad_ptr, x_grad_ptr, lod, width);
  } else if (pool_type == "SQRT") {
    lite::arm::math::seq_pool_sqrt_grad(
        din_ptr, dout_grad_ptr, x_grad_ptr, lod, width);
  } else if (pool_type == "MAX" || pool_type == "MIN") {
    lite::arm::math::seq_pool_max_grad(
        din_ptr, dout_grad_ptr, index_grad_ptr, x_grad_ptr, lod, width);
  } else if (pool_type == "FIRST") {
    lite::arm::math::seq_pool_first_grad(
        din_ptr, dout_grad_ptr, x_grad_ptr, lod, width);
  } else if (pool_type == "LAST") {
    lite::arm::math::seq_pool_last_grad(
        din_ptr, dout_grad_ptr, x_grad_ptr, lod, width);
  } else {
    LOG(ERROR) << " UNKNOWN sequence pool type";
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_pool_grad,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::SequencePoolGradCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Out@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("X@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("MaxIndex", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
