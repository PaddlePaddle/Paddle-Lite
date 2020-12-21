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

#include <cmath>
#include <string>
#include <vector>
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <PrecisionType Ptype, PrecisionType Otype>
class DepthwiseConv : public KernelLite<TARGET(kARM), Ptype> {
 public:
  typedef void (*conv_dw_impl)(const void* din,
                               void* dout,
                               int num,
                               int ch_out,
                               int h_out,
                               int w_out,
                               int ch_in,
                               int h_in,
                               int w_in,
                               const void* weights,
                               const float* bias,
                               const operators::ConvParam& param,
                               ARMContext* ctx,
                               const float* scale);
  DepthwiseConv() = default;
  ~DepthwiseConv() {}
  virtual void PrepareForRun();
  virtual void ReInitWhenNeeded();
  virtual void Run();

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
  }

  std::string kernel_func_name_{"NotImplForConvDw"};
#endif

 private:
  using param_t = operators::ConvParam;
  Tensor weights_;
  Tensor bias_;
  DDim last_shape_;
  bool flag_trans_weights_{false};
  bool flag_trans_bias_{false};
  conv_dw_impl impl_{nullptr};
  std::vector<float> w_scale_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
