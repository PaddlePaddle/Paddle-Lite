// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/arm/sparse_conv_compute.h"
#include <utility>
#include "lite/backends/arm/math/sparse_conv_impl.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <>
void SparseConvCompute<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {}

template <>
void SparseConvCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  const float* input = param.x->data<float>();
  const float* nonzero_weights = param.nonzero_weights->data<float>();
  const int32_t* diffs = param.diffs->data<int32_t>();
  const uint32_t* oc_nonzeros = param.oc_nonzeros->data<uint32_t>();
  const float* bias = param.bias ? param.bias->data<float>() : nullptr;
  float* dout = param.output->mutable_data<float>();

  auto x_dims = param.x->dims();
  auto o_dims = param.output->dims();
  int iw = x_dims[3];
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];
  int im_size = oh * ow;
  int first_ic = param.first_ic;
  const float* din = input + first_ic * im_size;
  lite::arm::math::sparse_conv_fp32_pipelined(nonzero_weights,
                                              din,
                                              diffs,
                                              oc_nonzeros,
                                              bias,
                                              dout,
                                              oc,
                                              ic,
                                              im_size,
                                              param,
                                              &ctx);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::arm::SparseConvCompute<PRECISION(kFloat),
                                                      PRECISION(kFloat)>
    SparseConvFp32;
REGISTER_LITE_KERNEL(sparse_conv2d, kARM, kFloat, kNCHW, SparseConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("NonZeroWeights", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("OcNonZeros", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Diffs", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
