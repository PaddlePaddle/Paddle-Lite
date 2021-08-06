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
#pragma once

#include <string>
#include <vector>
#include "lite/backends/x86/math/conv_utils.h"
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

template <PrecisionType Ptype, PrecisionType OutType>
class GemmLikeConv : public KernelLite<TARGET(kX86), Ptype> {
 public:
  GemmLikeConv() = default;
  ~GemmLikeConv() {}
  virtual void ReInitWhenNeeded() {
    auto& param = this->template Param<param_t>();
    CHECK(this->ctx_);
    auto& ctx = this->ctx_->template As<X86Context>();
    auto x_dims = param.x->dims();
    auto w_dims = param.filter->dims();
    auto o_dims = param.output->dims();
    if (last_shape_ == x_dims) {
      return;
    }

    int iw = x_dims[3];  // nchw
    int ih = x_dims[2];
    int ic = x_dims[1];
    int ow = o_dims[3];
    int oh = o_dims[2];
    int oc = o_dims[1];
    int kw = w_dims[3];
    int kh = w_dims[2];

    auto paddings = *param.paddings;
    auto dilations = *param.dilations;

    int sw = param.strides[1];
    int sh = param.strides[0];
    int pw = paddings[2];
    int ph = paddings[0];
    int dw = dilations[1];
    int dh = dilations[0];

    bool pads_equal =
        ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));

    int m = oc / param.groups;
    int k = ic * kh * kw / param.groups;
    int n = oh * ow;

    bool kps_equal = (pw == ph) && (sw == sh) && (kw == kh);
    bool ks_equal = (sw == sh) && (kw == kh);
    //! select conv gemmlike kernel
    if (kw == 1 && sw == 1 && pw == 0 && kps_equal && pads_equal) {
      //! 1x1s1p0 gemmlike conv
      flag_1x1gemm_ = true;
    } else {
      //! im2col gemmlike conv
      flag_1x1gemm_ = false;
    }
    // todo: GEMV support (n > 1 && m > 1)
    if (!flag_trans_weights_) {
      lite::x86::math::trans_gemm_weights<Ptype>(
          *(param.filter), weights_, param.groups, &ctx);
      flag_trans_weights_ = true;
    } else if (n == 1 || m == 1) {
      flag_trans_weights_ = false;
    }
    last_shape_ = x_dims;
  }
  virtual void PrepareForRun();
  virtual void Run();

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
  }

  std::string kernel_func_name_{"NotImplForConvDepthwise"};
#define PROFILE_INFO(dtype1, dtype2)                                        \
  template <>                                                               \
  void GemmLikeConv<PRECISION(dtype1), PRECISION(dtype2)>::                 \
      SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) { \
    ch->kernel_func_name = kernel_func_name_;                               \
  }

#define KERNEL_FUNC_NAME(kernel_func_name) kernel_func_name_ = kernel_func_name;

#else
#define PROFILE_INFO(dtype1, dtype2)
#define KERNEL_FUNC_NAME(kernel_func_name)
#endif

 private:
  using param_t = operators::ConvParam;
  DDim last_shape_;
  std::vector<float> w_scale_;
  bool flag_1x1gemm_{true};
  bool flag_trans_weights_{false};
  bool flag_trans_bias_{false};
  Tensor weights_;
  Tensor bias_;
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
