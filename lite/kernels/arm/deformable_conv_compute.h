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

#pragma once
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/kernel.h"
#include "lite/backends/arm/math/conv_impl.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <PrecisionType Ptype, PrecisionType OutType>
class DeformableConvCompute : public KernelLite<TARGET(kARM), Ptype> {
 public:
  virtual void PrepareForRun();

  virtual void ReInitWhenNeeded() {
    auto& param = this->Param<param_t>();
    auto w_dims = param.filter->dims();
    auto& ctx = this->ctx_->template As<ARMContext>();
    auto o_dims = param.output->dims();
    int n = o_dims[2] * o_dims[3];
    if (!flag_trans_weights_ && n > 1) {
        lite::arm::math::trans_gemm_weights<Ptype>(
            *(param.filter), weights_, param.groups, &ctx);
        flag_trans_weights_ = true;
    } else {
        flag_trans_weights_ = false;
    }
  }

  virtual void Run();

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    impl_->SetProfileRuntimeKernelInfo(ch);
  }
#endif

  ~DeformableConvCompute() {
    if (impl_ != nullptr) {
      delete impl_;
    }
  }

 private:
  using param_t = operators::DeformableConvParam;
  bool flag_trans_weights_;
  Tensor weights_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
