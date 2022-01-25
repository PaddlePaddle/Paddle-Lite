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
#include "lite/backends/x86/math/avx/conv_utils.h"
#include "lite/backends/x86/math/conv_direct_fp32.h"
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

#define ROUNDUP(a, b) ((((a) + (b)-1) / (b)) * (b))

// only support 3x3s2
template <PrecisionType Ptype, PrecisionType OutType>
class DirectConv : public KernelLite<TARGET(kX86), Ptype> {
 public:
  DirectConv() = default;
  ~DirectConv() { delete code_; }

  virtual void Run();

  virtual void PrepareForRun() {
    auto& param = this->template Param<param_t>();
#ifdef __AVX__
    constexpr int block = 8;
#else
    constexpr int block = 4;
#endif

    int oc = param.filter->dims()[0];
    int ic = param.filter->dims()[1];
    int wh = param.filter->dims()[2];
    int ww = param.filter->dims()[3];
    int cround = ROUNDUP(oc, block);
    oc_expand_ = cround;
    // [chout, chin, wh, ww] -> [chout / block, chin, wh, ww, block]
    weights_.Resize({cround / block, ic, wh, ww, block});
    auto filter_data = param.filter->template data<float>();
    auto weights_w_data = weights_.mutable_data<float>();
    lite::x86::math::conv_trans_weights_numc(
        filter_data, weights_w_data, oc, ic, wh, ww, block);

    auto x_dims = param.x->dims();
    auto w_dims = param.filter->dims();
    auto o_dims = param.output->dims();

    const int ph = (*(param.paddings))[0];
    const int pw = (*(param.paddings))[2];

    int iw = x_dims[3];
    int ih = x_dims[2];
    int oh = o_dims[2];
    int ow = o_dims[3];
    code_ = new lite::x86::math::conv_direct();
    code_->generate_code(
        ic, ih, iw, oc, oc_expand_, oh, ow, ph, pw, wh, ww, param.strides[1]);
    code_->ready();
  }

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
  }

  std::string kernel_func_name_{"NotImplForConvDirect"};
#endif

 private:
  using param_t = operators::ConvParam;
  Tensor weights_;
  Tensor bias_;
  Tensor trans_in_;
  bool flag_trans_weights_{false};
  bool flag_trans_bias_{false};
  std::vector<float> w_scale_;
  int oc_expand_;
  lite::x86::math::conv_direct* code_;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
