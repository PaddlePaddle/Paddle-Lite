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

#include "lite/kernels/intel_fpga/conv_depthwise.h"
#include "lite/backends/arm/math/conv_block_utils.h"
#include "lite/backends/arm/math/conv_impl.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace intel_fpga {

template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::ReInitWhenNeeded() {}

template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<ARMContext>();
  auto w_dims = param.filter->dims();
  auto kw = w_dims[3];
  auto channel = w_dims[0];
  auto hin = param.x->dims()[2];
  auto win = param.x->dims()[3];
  auto paddings = *param.paddings;
  // select dw conv kernel
  if (kw == 3) {
    bool pads_less = ((paddings[1] < 2) && (paddings[3] < 2));
    if (pads_less && paddings[0] == paddings[2] &&
        (paddings[0] == 0 || paddings[0] == 1)) {
      flag_trans_weights_ = false;
    } else {
      // trans weights
      constexpr int cblock = 4;
      auto oc = w_dims[0];
      auto kh = w_dims[2];
      auto cround = ROUNDUP(oc, cblock);
      weights_.Resize({cround, 1, kh, kw});
      auto w_data = weights_.mutable_data<float>();
      auto w_data_in = param.filter->data<float>();
      lite::arm::math::conv_trans_weights_numc(
          w_data_in, w_data, oc, 1, cblock, kh * kw);
      flag_trans_weights_ = true;
    }
    impl_ = lite::arm::math::conv_depthwise_3x3_fp32;
  } else if (kw == 5) {
    auto strides = param.strides;
    if ((strides[0] == 1 && strides[1] == 1) ||
        (strides[0] == 2 && strides[1] == 2)) {
      // trans weights
      constexpr int cblock = 4;
      auto oc = w_dims[0];
      auto kh = w_dims[2];
      auto cround = ROUNDUP(oc, cblock);
      weights_.Resize({cround, 1, kh, kw});
      auto w_data = weights_.mutable_data<float>();
      auto w_data_in = param.filter->data<float>();
      lite::arm::math::conv_trans_weights_numc(
          w_data_in, w_data, oc, 1, cblock, kh * kw);
      flag_trans_weights_ = true;
      impl_ = lite::arm::math::conv_depthwise_5x5_fp32;
    } else {
      LOG(FATAL)
          << "5x5 depthwise conv only support stride == 1 or stride == 2";
    }
  } else {
    LOG(FATAL) << "this type dw conv not impl";
  }
}

template <>
void DepthwiseConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  CHECK(this->ctx_);
  auto& ctx = this->ctx_->template As<ARMContext>();
  const auto* i_data = param.x->data<float>();
  const auto* w_data = flag_trans_weights_ ? weights_.data<float>()
                                           : param.filter->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  if (flag_trans_bias_) {
    b_data = bias_.data<float>();
  }
  auto* o_data = param.output->mutable_data<float>();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];  // nchw
  int ih = x_dims[2];
  int ic = x_dims[1];
  int bs = x_dims[0];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int oc = o_dims[1];

  impl_(i_data,
        o_data,
        bs,
        oc,
        oh,
        ow,
        ic,
        ih,
        iw,
        w_data,
        b_data,
        param,
        &ctx,
        w_scale_.data());
}

}  // namespace intel_fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
