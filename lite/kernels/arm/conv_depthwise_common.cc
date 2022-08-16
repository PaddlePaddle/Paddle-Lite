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

#include "lite/kernels/arm/conv_depthwise_common.h"
#include "lite/backends/arm/math/conv_block_utils.h"
#include "lite/backends/arm/math/conv_impl.h"
#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/conv_impl_fp16.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <>
void DepthwiseConvCommon<PRECISION(kFloat),
                         PRECISION(kFloat)>::ReInitWhenNeeded() {}

template <>
void DepthwiseConvCommon<PRECISION(kFloat),
                         PRECISION(kFloat)>::PrepareForRun() {}

template <>
void DepthwiseConvCommon<PRECISION(kInt8),
                         PRECISION(kFloat)>::ReInitWhenNeeded() {}

template <>
void DepthwiseConvCommon<PRECISION(kInt8), PRECISION(kFloat)>::PrepareForRun() {
}

template <>
void DepthwiseConvCommon<PRECISION(kInt8),
                         PRECISION(kInt8)>::ReInitWhenNeeded() {}

template <>
void DepthwiseConvCommon<PRECISION(kInt8), PRECISION(kInt8)>::PrepareForRun() {}

template <>
void DepthwiseConvCommon<PRECISION(kFloat), PRECISION(kFloat)>::Run() {}

template <>
void DepthwiseConvCommon<PRECISION(kInt8), PRECISION(kFloat)>::Run() {}

template <>
void DepthwiseConvCommon<PRECISION(kInt8), PRECISION(kInt8)>::Run() {}

#ifdef ENABLE_ARM_FP16
template <>
void DepthwiseConvCommon<PRECISION(kFP16),
                         PRECISION(kFP16)>::ReInitWhenNeeded() {}

template <>
void DepthwiseConvCommon<PRECISION(kFP16), PRECISION(kFP16)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto w_dims = param.filter->dims();
  auto kh = w_dims[2];
  auto kw = w_dims[3];
  auto oc = w_dims[0];
  constexpr int cblock = 8;
  auto cround = ROUNDUP(oc, cblock);
  weights_.Resize({cround, 1, kh, kw});
  auto w_data = weights_.mutable_data<float16_t>();
  auto w_data_in = param.filter->data<float16_t>();
  lite::arm::math::conv_trans_weights_numc(
      w_data_in, w_data, oc, 1, cblock, kh * kw);
  KERNEL_FUNC_NAME("conv_depthwise_common_fp16")
}
PROFILE_INFO(kFP16, kFP16)

template <>
void DepthwiseConvCommon<PRECISION(kFP16), PRECISION(kFP16)>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<ARMContext>();
  lite::arm::math::fp16::conv_depthwise_common(
      weights_.data<float16_t>(), param, &ctx);
}

#endif
}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
