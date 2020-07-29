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
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/kernel.h"
#ifdef LITE_WITH_PROFILE
#include <string>
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
    auto& param = this->template Param<param_t>();
    auto& x_dims = param.x->dims();
    auto w_dims = param.conv_param.filter->dims();
    auto& ctx = this->ctx_->template As<ARMContext>();
    auto o_dims = param.output->dims();
    int n = o_dims[2] * o_dims[3];
    if (last_shape_ == x_dims && last_weights_shape_ == w_dims) {
      return;
    }
    if (n > 1) {
      lite::arm::math::trans_gemm_weights<Ptype>(
          *(param.conv_param.filter), weights_, param.conv_param.groups, &ctx);
      flag_trans_weights_ = true;
    } else if (n == 1) {
      flag_trans_weights_ = false;
    }
    last_shape_ = x_dims;
    last_weights_shape_ = w_dims;
  }

  virtual void Run();

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
  }
  std::string kernel_func_name_{"NotImplForDeformableConv"};
#endif

  ~DeformableConvCompute() = default;

 private:
  using param_t = operators::DeformableConvParam;
  DDim last_shape_;
  DDim last_weights_shape_;
  bool flag_trans_weights_;
  Tensor weights_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
