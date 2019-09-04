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
#include <vector>
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <PrecisionType Ptype, PrecisionType Otype>
class GemmLikeConv : public KernelLite<TARGET(kARM), Ptype> {
 public:
  GemmLikeConv() = default;
  ~GemmLikeConv() {}

  virtual void Init() {
    auto& param = this->template Param<param_t>();
    CHECK(this->ctx_);
    auto& ctx = this->ctx_->template As<ARMContext>();
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
    int sw = param.strides[1];
    int sh = param.strides[0];
    int pw = param.paddings[1];
    int ph = param.paddings[0];
    int dw = param.dilations[1];
    int dh = param.dilations[0];

    int m = oc / param.groups;
    int k = ic * kh * kw / param.groups;
    int n = oh * ow;

    bool kps_equal = (pw == ph) && (sw == sh) && (kw == kh);
    bool ks_equal = (sw == sh) && (kw == kh);
    //! select conv gemmlike kernel
    if (kw == 1 && sw == 1 && pw == 0 && kps_equal) {
      //! 1x1s1p0 gemmlike conv
      flag_1x1gemm_ = true;
    } else {
      //! im2col gemmlike conv
      flag_1x1gemm_ = false;
      ctx.ExtendWorkspace(k * n * sizeof(float));
    }
    if (!flag_trans_weights_ && n > 1) {
      lite::arm::math::trans_gemm_weights<Ptype>(
          *(param.filter), weights_, param.groups, &ctx);
      flag_trans_weights_ = true;
    }
    last_shape_ = x_dims;
  }

  virtual void PrepareForRun() {
    LOG(FATAL) << "GemmLikeConv PrepareForRun not implemented";
  }

  virtual void Run() { LOG(FATAL) << "GemmLikeConv Run not implemented"; }

  /// todo, support inplace weights transform
 protected:
  using param_t = operators::ConvParam;
  DDim last_shape_;
  std::vector<float> w_scale_;
  bool flag_1x1gemm_{true};
  bool flag_trans_weights_{false};
  bool flag_trans_bias_{false};
  Tensor weights_;
  Tensor bias_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
