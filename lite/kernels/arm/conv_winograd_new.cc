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

#include "lite/kernels/arm/conv_winograd_new.h"
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

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();

  if (last_shape_ == x_dims) {
    return;
  }

  int ic = x_dims[1];
  int ow = o_dims[3];
  int oh = o_dims[2];
  int oc = o_dims[1];
  int tile_w = (ow + 5) / 6;
  int tile_h = (oh + 5) / 6;

  int tile_block = 8;
#ifdef __aarch64__
  tile_block = 16;
#endif

  int threads = ctx.threads();
  int need_tmp_size = tile_block * (ic + oc) * 64 + threads * 128;
  ctx.ExtendWorkspace(need_tmp_size * sizeof(float));
  last_shape_ = x_dims;
}

template <>
void WinogradConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<ARMContext>();

  auto x_dims = param.x->dims();
  auto w_dims = param.filter->dims();
  auto o_dims = param.output->dims();
  last_shape_ = x_dims;

  int ic = x_dims[1];
  int ow = o_dims[3];
  int oh = o_dims[2];
  int oc = o_dims[1];
  int tile_w = (ow + 5) / 6;
  int tile_h = (oh + 5) / 6;

  int tile_block = 8;
#ifdef __aarch64__
  tile_block = 16;
#endif

  int threads = ctx.threads();
  int need_tmp_size = tile_block * (ic + oc) * 64 + threads * 128;
  ctx.ExtendWorkspace(need_tmp_size * sizeof(float));

  int w_h = (oc + 3) / 4 * 4;
  weights_.Resize({64, w_h, ic});

  auto weights_wino =
      static_cast<float*>(malloc(sizeof(float) * 8 * 8 * w_h * ic));
  void* trans_tmp_ptr = malloc(sizeof(float) * 8 * 8 * oc * ic);
  lite::arm::math::weight_trans(
      weights_wino, param.filter->data<float>(), oc, ic, trans_tmp_ptr);
  auto weights_trans = weights_.mutable_data<float>();

  memcpy(weights_trans, weights_wino, sizeof(float) * 8 * 8 * w_h * ic);

  free(trans_tmp_ptr);
  free(weights_wino);
}

template <>
void WinogradConv<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
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

  lite::arm::math::conv_compute_6x6_3x3_0(
      i_data, o_data, bs, oc, oh, ow, ic, ih, iw, w_data, b_data, param, &ctx);
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
