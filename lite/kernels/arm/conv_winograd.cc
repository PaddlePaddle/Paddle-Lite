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

  int ic = x_dims[1];
  int ih = x_dims[2];
  int iw = x_dims[3];
  int oc = o_dims[1];
  int pad_h = param.paddings[0];
  int pad_w = param.paddings[1];
  int tile_block = 8;
#ifdef __aarch64__
  tile_block = 16;
#endif
  const int new_input_size =
      (ic + 3) / 4 * 4 * (ih + pad_h * 2) * (iw + pad_w * 2);
  const int temp_size =
      (tile_block * ((ic + 3) / 4 + (oc + 3) / 4) * 256 + 512) * threads;
  ctx.ExtendWorkspace((temp_size + new_input_size) * sizeof(float));
  last_shape_ = x_dims;
}

template <>
void WinogradConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  int threads = ctx.threads();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();
  last_shape_ = x_dims;

  int ic = x_dims[1];
  int ih = x_dims[2];
  int iw = x_dims[3];
  int oc = o_dims[1];
  int pad_h = param.paddings[0];
  int pad_w = param.paddings[1];
  int oc_pad = (oc + 3) / 4 * 4;
  int ic_pad = (ic + 3) / 4 * 4;
  int tile_block = 8;
#ifdef __aarch64__
  tile_block = 16;
#endif
  const int new_input_size =
      (ic + 3) / 4 * 4 * (ih + pad_h * 2) * (iw + pad_w * 2);
  const int temp_size =
      (tile_block * ((ic + 3) / 4 + (oc + 3) / 4) * 256 + 512) * threads;

  weights_.Resize({1, 1, 1, 64 * oc_pad * ic_pad});
  ctx.ExtendWorkspace((temp_size + new_input_size) * sizeof(float));
  void* trans_tmp_ptr = malloc(sizeof(float) * 8 * 8 * oc * ic);
  auto weights_data_ = weights_.mutable_data<float>();
  lite::arm::math::weight_trans_c4(
      weights_data_, param.filter->data<float>(), ic, oc, trans_tmp_ptr);
  free(trans_tmp_ptr);
}

template <>
void WinogradConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  LOG(INFO) << "winograd";
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();
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
  lite::arm::math::conv_compute_6x6_3x3(
      i_data, o_data, bs, oc, oh, ow, ic, ih, iw, w_data, b_data, param, &ctx);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
