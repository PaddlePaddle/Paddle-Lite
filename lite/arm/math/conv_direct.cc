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

#include "lite/arm/math/conv_direct.h"
#include "lite/arm/math/conv_block_utils.h"
#include "lite/arm/math/conv_impl.h"

namespace paddle {
namespace lite {
namespace arm {
namespace math {

template <>
bool DirectConv<PRECISION(kFloat)>::create(const operators::ConvParam& param,
                                           ARMContext* ctx) {
  this->ctx_ = ctx;
  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];  // nchw
  int ic = x_dims[1];
  int ow = o_dims[3];
  int oc = o_dims[1];
  int kw = w_dims[3];
  int sw = param.strides[1];
  // select dw conv kernel
  const auto* w_data = param.filter->data<float>();
  if (kw == 3 && sw == 1) {
    VLOG(5) << "invoke 3x3s1 direct conv";
    impl_ = conv_3x3s1_direct_fp32;

    constexpr int cblock = 4;
    int cround = (oc + cblock - 1) / cblock * cblock;
    weights_trans_.Resize({cround, ic, kw, kw});
    float* transed_w_data = weights_trans_.mutable_data<float>();

    conv_trans_weights_numc(w_data, transed_w_data, oc, ic, cblock, kw * kw);
    is_weights_transed_ = true;
  } else if (kw == 3 && sw == 2) {
    VLOG(5) << "invoke 3x3s2 direct conv";
    impl_ = conv_3x3s2_direct_fp32;

    constexpr int cblock = 4;
    int cround = (oc + cblock - 1) / cblock * cblock;
    weights_trans_.Resize({cround, ic, kw, kw});
    float* transed_w_data = weights_trans_.mutable_data<float>();
    conv_trans_weights_numc(w_data, transed_w_data, oc, ic, cblock, kw * kw);
    is_weights_transed_ = true;
  } else {
    LOG(ERROR) << "this type direct conv not impl";
    return false;
  }
  return true;
}

template <>
bool DirectConv<PRECISION(kFloat)>::init(const operators::ConvParam& param,
                                         Context<TARGET(kARM)>* ctx) {
  this->ctx_ = ctx;
  return create(param, ctx);
}

template <>
bool DirectConv<PRECISION(kFloat)>::run(const operators::ConvParam& param) {
  // start timer
  const auto* i_data = param.x->data<float>();
  const auto* w_data = param.filter->data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
  auto* o_data = param.output->mutable_data<float>();

  if (is_weights_transed_ == true) {
    w_data = weights_trans_.data<float>();
  }
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
        this->ctx_);

  // timer end
  return true;
}

template <PrecisionType Ptype_out>
bool DirectConvInt8<Ptype_out>::create(const operators::ConvParam& param,
                                       ARMContext* ctx) {
  this->ctx_ = ctx;
  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  int iw = x_dims[3];  // nchw
  int ic = x_dims[1];
  int ow = o_dims[3];
  int oc = o_dims[1];
  int kw = w_dims[3];
  int sw = param.strides[1];
  // select dw conv kernel
  w_scale_ = param.weight_scale;
  //! update weights scale
  const auto* w_data = param.filter->data<int8_t>();
  if (Ptype_out == PRECISION(kInt8) || Ptype_out == PRECISION(kFloat)) {
    CHECK_EQ(this->w_scale_.size(), oc) << "weights scale size must be chout";
    float input_scale = param.input_scale;
    for (auto& w_s : w_scale_) {
      w_s *= input_scale;
      if (Ptype_out == PRECISION(kInt8)) {
        w_s /= param.output_scale;
      }
    }
  }
  if (kw == 3 && sw == 1) {
    VLOG(5) << "invoke 3x3s1 direct conv";
    impl_int8_ = conv_3x3s1_direct_int8;

    constexpr int cblock = 4;
    int inpad = 4;
    int cround = (oc + cblock - 1) / cblock * cblock;
    weights_trans_.Resize({cround, ic, kw, kw});
    int8_t* transed_w_data = weights_trans_.mutable_data<int8_t>();
    conv_trans_weights_numc(w_data, transed_w_data, oc, ic, cblock, kw * kw);

    int wout_round = ((ow + 3) / 4) * 4;
    int win_round = wout_round * sw + inpad;
    int row_out = 2;
    int row_in = 4;
    int tmp_size_out = wout_round * row_out * cblock;
    int in_len = win_round * ic;
    int tmp_size_in = row_in * in_len;
    ctx_->ExtendWorkspace(ctx_->threads() * tmp_size_out +
                          (tmp_size_in + 3) / 4 * 4 + wout_round + win_round);
    is_weights_transed_ = true;

  } else if (kw == 3 && sw == 2) {
    VLOG(5) << "invoke 3x3s2 direct conv";
    impl_int8_ = conv_3x3s2_direct_int8;

    // constexpr int cblock = 4;
    int cblock = conv_3x3s2_direct_int8_c_num();
    int cround = (oc + cblock - 1) / cblock * cblock;
    weights_trans_.Resize({cround, ic, kw, kw});
    int8_t* transed_w_data = weights_trans_.mutable_data<int8_t>();
    conv_trans_weights_numc(w_data, transed_w_data, oc, ic, cblock, kw * kw);
    is_weights_transed_ = true;

  } else {
    LOG(ERROR) << "this type direct conv not impl";
    return false;
  }
  return true;
}

template <PrecisionType Ptype_out>
bool DirectConvInt8<Ptype_out>::init(const operators::ConvParam& param,
                                     Context<TARGET(kARM)>* ctx) {
  this->ctx_ = ctx;
  return create(param, ctx);
}

template <PrecisionType Ptype_out>
bool DirectConvInt8<Ptype_out>::run(const operators::ConvParam& param) {
  // start timer
  const auto* i_data = param.x->data<int8_t>();
  const auto* w_data = param.filter->data<int8_t>();
  const auto* b_data = param.bias ? param.bias->data<int32_t>() : nullptr;
  auto* o_data = param.output->mutable_data<int32_t>();
  if (is_weights_transed_ == true) {
    w_data = weights_trans_.data<int8_t>();
  }
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

  impl_int8_(i_data,
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
             this->ctx_,
             Ptype_out,
             w_scale_.data());

  // Modified from int32 for debug convenience
  if (Ptype_out == PRECISION(kInt8)) param.output->mutable_data<int8_t>();
  return true;
}

template class DirectConvInt8<PRECISION(kInt8)>;
template class DirectConvInt8<PRECISION(kFloat)>;
template class DirectConvInt8<PRECISION(kInt32)>;

}  // namespace math
}  // namespace arm
}  // namespace lite
}  // namespace paddle
