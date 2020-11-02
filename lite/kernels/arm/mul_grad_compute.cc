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

#include "lite/kernels/arm/mul_grad_compute.h"
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/backends/arm/math/sgemm.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void MulGradCompute::PrepareForRun() {
  auto& ctx = this->ctx_->template As<ARMContext>();
}

void MulGradCompute::Run() {
  // step1 flatten_2d
  auto& param = Param<param_t>();
  const auto x_dims = param.x->dims();
  const auto y_dims = param.y->dims();
  const auto out_dims = param.output_grad->dims();

  m_ = static_cast<int>(x_dims.Slice(0, param.x_num_col_dims).production());

  k_ = static_cast<int>(
      x_dims.Slice(param.x_num_col_dims, x_dims.size()).production());
  n_ = static_cast<int>(
      y_dims.Slice(param.y_num_col_dims, y_dims.size()).production());

  const auto* out_grad_data = param.output_grad->data<float>();
  const auto* x_data = param.x->data<float>();
  const auto* y_data = param.y->data<float>();
  float* x_grad_data;
  float* y_grad_data;
  if (param.x_grad) {
    x_grad_data = param.x_grad->mutable_data<float>();
  }

  if (param.y_grad) {
    y_grad_data = param.y_grad->mutable_data<float>();
  }

  paddle::lite::operators::ActivationParam act_param;
  act_param.has_active = false;
  // out_grad  * y^T = x_grad
  // (m, n), (n, k) -> (m, k)
  auto& ctx = this->ctx_->template As<ARMContext>();
  if (param.x_grad) {
    if (m_ == 1) {
      lite::arm::math::sgemv(y_data,
                             out_grad_data,
                             x_grad_data,
                             false,
                             k_,  // M
                             n_,  // N
                             0.f,
                             false,
                             nullptr,
                             false,
                             lite_api::ActivationType::kIndentity,
                             &ctx);
    } else {
      paddle::lite::arm::math::sgemm(false,
                                     true,           // is_transB,
                                     m_,             // M
                                     k_,             // N
                                     n_,             // K
                                     1.0f,           // alpha
                                     out_grad_data,  // A
                                     n_,             // lda
                                     y_data,         // B
                                     n_,             // ldb
                                     0.f,            // beta
                                     x_grad_data,    // C
                                     k_,             // ldc
                                     NULL,           // bias
                                     false,          // is_bias
                                     act_param,      // act_param
                                     &ctx);          // ctx
    }
  }

  // x^T * out_grad = y_grad
  // (k, m) (m, n) -> (k, n)
  if (param.y_grad) {
    if (n_ == 1) {
      lite::arm::math::sgemv(x_data,
                             out_grad_data,
                             y_grad_data,
                             true,
                             k_,  // M
                             m_,  // N
                             0.f,
                             false,
                             nullptr,
                             false,
                             lite_api::ActivationType::kIndentity,
                             &ctx);
    } else {
      paddle::lite::arm::math::sgemm(true,           // is_transA
                                     false,          // is_transB,
                                     k_,             // M
                                     n_,             // N
                                     m_,             // K
                                     1.0f,           // alpha
                                     x_data,         // A
                                     k_,             // lda
                                     out_grad_data,  // B
                                     n_,             // ldb
                                     0.f,            // beta
                                     y_grad_data,    // C
                                     n_,             // ldc
                                     NULL,           // bias
                                     false,          // is_bias
                                     act_param,      // act_param
                                     &ctx);          // ctx
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(mul_grad,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::MulGradCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Out@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("X@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y@GRAD", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
