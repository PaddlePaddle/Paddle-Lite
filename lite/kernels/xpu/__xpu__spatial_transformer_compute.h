// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#define USE_XFT

#ifdef USE_XFT
#include "layers/spatial_transformer.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

namespace xft = baidu::xpu::xft;

template <typename T>
struct identity {
  typedef T type;
};

template <typename InType, PrecisionType PType>
class XPUSpatialTransformerCompute : public KernelLite<TARGET(kXPU), PType> {
 public:
  using param_t = operators::XPUSpatialTransformerParam;

  virtual void PrepareForRun();

  virtual void Run();

  virtual ~XPUSpatialTransformerCompute() = default;

 private:
#ifdef USE_XFT
  xft::SpatialTransformerFusionParam st_param;
  std::vector<xft::xftVec<float>> xft_gn_weights;
  std::vector<xft::xftVec<float>> xft_gn_bias;
  std::vector<xft::xftVec<float>> xft_ln_weights;
  std::vector<xft::xftVec<float>> xft_ln_bias;
  std::vector<xft::xftMat<int16_t>> xft_q_weights;
  std::vector<xft::xftMat<int16_t>> xft_k_weights;
  std::vector<xft::xftMat<int16_t>> xft_v_weights;
  std::vector<xft::xftMat<int16_t>> xft_attn_fc_weights;
  std::vector<xft::xftVec<float>> xft_attn_fc_bias;
  std::vector<xft::xftMat<int16_t>> xft_geglu_fc_weights;
  std::vector<xft::xftVec<float>> xft_geglu_fc_bias;
  std::vector<xft::xftTensor<int16_t, 4>> xft_conv_weights;
  std::vector<xft::xftVec<float>> xft_conv_bias;
#else
  std::vector<const float *> arg_fc_bias_;
  std::vector<const float *> arg_ln_scale_;
  std::vector<const float *> arg_ln_bias_;
  std::vector<const float *> arg_gn_scale_;
  std::vector<const float *> arg_gn_bias_;
  std::vector<const float *> arg_conv_bias_;
#endif
  std::vector<const int16_t *> arg_fc_weight_int16_;
  std::vector<const int16_t *> arg_conv_filter_int16_;
  std::vector<const float *> fc_weight_max_;
  std::vector<const float *> conv_filter_max_;
  XPUScratchPadGuard weight_max_guard_;
  XPUScratchPadGuard filter_max_guard_;

  template <typename T>
  std::vector<const T *> *GetWeight() {
    LOG(FATAL) << "Invalid Weight Type";
    return nullptr;
  }

  std::vector<const int16_t *> *GetWeight() { return &arg_fc_weight_int16_; }

  template <typename T>
  std::vector<const T *> *GetFilter() {
    LOG(FATAL) << "Invalid Weight Type";
    return nullptr;
  }

  std::vector<const int16_t *> *GetFilter() { return &arg_conv_filter_int16_; }

  void PrepareWeightMax(const std::vector<lite::Tensor *> &weight_max,
                        int max_ptr_len,
                        std::vector<const float *> *max_xpu_ptrs);
  void PrepareFilterMax(const std::vector<lite::Tensor *> &filter_max,
                        int max_ptr_len,
                        std::vector<const float *> *max_xpu_ptrs);
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
