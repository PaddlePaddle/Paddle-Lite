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

#include "lite/kernels/arm/conv_winograd.h"
#include <vector>
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/backends/arm/math/packed_sgemm.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <>
void WinogradConv<PRECISION(kFloat), PRECISION(kFloat)>::ReInitWhenNeeded() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  int threads = ctx.threads();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  if (last_shape_ == x_dims) {
    return;
  }
  last_shape_ = x_dims;
  //! update workspace size
  int ic = x_dims[1];
  int ih = x_dims[2];
  int iw = x_dims[3];
  int oc = o_dims[1];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int tile_block = 8;
  auto pad = *(param.paddings);
  int pad_h0 = pad[0];
  int pad_h1 = pad[1];
  int pad_w0 = pad[2];
  int pad_w1 = pad[3];
  int oc_pad = (oc + 3) / 4 * 4;
  int ic_pad = (ic + 3) / 4 * 4;
  const int new_input_size =
      (ic + 3) / 4 * 4 * (ih + pad_h0 + pad_h1) * (iw + pad_w0 + pad_w1);
  const int temp_size =
      (tile_block * ((ic + 3) / 4 + (oc + 3) / 4) * 4 * wino_iw * wino_iw +
       8 * wino_iw * wino_iw) *
      threads;
  workspace_size_ = (temp_size + new_input_size) * sizeof(float);

  //! update trans weights impl
  choose_small_ = ow * oh / (tile_block * threads) < 36 ? true : false;
  if (choose_small_) {
    wino_iw = 4;

    if (last_function_ == 0) {
      return;
    }
    last_function_ = 0;
  } else {
    wino_iw = 8;
    if (last_function_ == 1) {
      return;
    }
    last_function_ = 1;
  }

  weights_.Resize({1, 1, 1, wino_iw * wino_iw * oc_pad * ic_pad});
  void* trans_tmp_ptr = malloc(sizeof(float) * wino_iw * wino_iw * oc * ic);
  auto weights_data_ = weights_.mutable_data<float>();
  if (!choose_small_) {
    lite::arm::math::weight_trans_c4_8x8(
        weights_data_, param.filter->data<float>(), ic, oc, trans_tmp_ptr);
  } else {
    lite::arm::math::weight_trans_c4_4x4(
        weights_data_, param.filter->data<float>(), ic, oc, trans_tmp_ptr);
  }
  free(trans_tmp_ptr);
}

template <>
void WinogradConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
}

template <>
void WinogradConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  ctx.ExtendWorkspace(workspace_size_);
  const auto* i_data = param.x->data<float>();
  const auto* w_data = weights_.data<float>();
  const auto* b_data = param.bias ? param.bias->data<float>() : nullptr;
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

  if (!choose_small_) {
    lite::arm::math::conv_compute_6x6_3x3(i_data,
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
                                          &ctx);
  } else {
    int tile_block = 8;
    int block_count =
        (((ow + 1) / 2) * ((oh + 1) / 2) + tile_block - 1) / tile_block;
    if (block_count != 1) {
      lite::arm::math::conv_compute_2x2_3x3(i_data,
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
                                            &ctx);
    } else {
      lite::arm::math::conv_compute_2x2_3x3_small(i_data,
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
                                                  &ctx);
    }
  }
  ReInitWhenNeeded();
}

template <PrecisionType OutType>
void WinogradConv<PRECISION(kInt8), OutType>::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  ctx.ExtendWorkspace(workspace_size_);
  const auto* i_data = param.x->template data<int8_t>();
  const auto* w_data = weights_.data<int16_t>();
  const auto* b_data = param.bias ? bias_.data<float>() : nullptr;
  // const float* i_data;
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

  // now  always choose small
  if (OutType == PRECISION(kInt8)) {
    auto* o_data = param.output->template mutable_data<int8_t>();
    lite::arm::math::conv_compute_2x2_3x3_int8<int8_t>(i_data,
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
                                                       w_scale_.data(),
                                                       param,
                                                       &ctx);
  } else {
    auto* o_data = param.output->template mutable_data<float>();
    lite::arm::math::conv_compute_2x2_3x3_int8<float>(i_data,
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
                                                      w_scale_.data(),
                                                      param,
                                                      &ctx);
  }
#ifdef LITE_WITH_PROFILE
  kernel_func_name_ = "conv_compute_2x2_3x3_int8";
#endif
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
