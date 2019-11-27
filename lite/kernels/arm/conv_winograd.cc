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

  int ic = x_dims[1];
  int ih = x_dims[2];
  int iw = x_dims[3];
  int oc = o_dims[1];
  int oh = o_dims[2];
  int ow = o_dims[3];
  int tile_block = 8;
#ifdef __aarch64__
  tile_block = 16;
#endif
  int parallel_threads =
      (((ow + 5) / 6) * ((oh + 5) / 6) + tile_block - 1) / tile_block;
  if (threads <= 2 && parallel_threads >= threads) {
    if (last_kernel_is_c4_ == 1) {
      return;
    }
    last_kernel_is_c4_ = 1;
    auto pad = *(param.paddings);
    int pad_h = pad[0];
    int pad_w = pad[2];
    int oc_pad = (oc + 3) / 4 * 4;
    int ic_pad = (ic + 3) / 4 * 4;
    const int new_input_size =
        (ic + 3) / 4 * 4 * (ih + pad_h * 2) * (iw + pad_w * 2);
    const int temp_size =
        (tile_block * ((ic + 3) / 4 + (oc + 3) / 4) * 256 + 512) * threads;
    ctx.ExtendWorkspace((temp_size + new_input_size) * sizeof(float));

    weights_.Resize({1, 1, 1, 64 * oc_pad * ic_pad});
    ctx.ExtendWorkspace((temp_size + new_input_size) * sizeof(float));
    void* trans_tmp_ptr = malloc(sizeof(float) * 8 * 8 * oc * ic);
    auto weights_data_ = weights_.mutable_data<float>();
    lite::arm::math::weight_trans_c4(
        weights_data_, param.filter->data<float>(), ic, oc, trans_tmp_ptr);
    free(trans_tmp_ptr);
  } else {
    if (last_kernel_is_c4_ == 0) {
      return;
    }
    last_kernel_is_c4_ = 0;
    int tile_w = (ow + 5) / 6;
    int tile_h = (oh + 5) / 6;

    int size_tile = tile_h * tile_w;
    int size_trans_channel = 8 * 8 * size_tile;
    int max_ch = ic > oc ? ic : oc;

    const int n_wino = size_tile;
    ctx.ExtendWorkspace((size_trans_channel * max_ch * 2 + n_wino) *
                        sizeof(float));

    const int m_wino = oc;
    int hblock = lite::arm::math::get_hblock(&ctx);
    int m_round = hblock * ((m_wino + hblock - 1) / hblock);
    weights_.Resize({1, 1, 1, 8 * 8 * m_round * ic});
    ctx.ExtendWorkspace((size_trans_channel * max_ch * 2 + n_wino) *
                        sizeof(float));
    auto weights_wino =
        static_cast<float*>(malloc(sizeof(float) * 8 * 8 * oc * ic));
    void* trans_tmp_ptr = malloc(sizeof(float) * 8 * 8 * oc * ic);
    lite::arm::math::winograd_transform_weights(
        weights_wino, param.filter->data<float>(), oc, ic, trans_tmp_ptr);
    auto weights_trans = weights_.mutable_data<float>();
    for (int i = 0; i < 64; ++i) {
      float* packed_weights = weights_trans + i * m_round * ic;
      const float* weights_wino_ptr = weights_wino + i * oc * ic;
      lite::arm::math::prepackA(packed_weights,
                                weights_wino_ptr,
                                1.f,
                                ic,
                                0,
                                m_wino,
                                0,
                                ic,
                                false,
                                &ctx);
    }
    free(trans_tmp_ptr);
    free(weights_wino);
  }
  last_shape_ = x_dims;
}

template <>
void WinogradConv<PRECISION(kFloat), PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
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

  int tile_block = 8;
#ifdef __aarch64__
  tile_block = 16;
#endif
  int threads = ctx.threads();
  int parallel_threads =
      (((ow + 5) / 6) * ((oh + 5) / 6) + tile_block - 1) / tile_block;
  if (threads <= 2 && parallel_threads >= threads) {
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
    lite::arm::math::conv_winograd3x3(i_data,
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

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
