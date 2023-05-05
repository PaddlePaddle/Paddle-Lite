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
#include "layers/vae_decoder.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

namespace xft = baidu::xpu::xft;

template <typename InType, PrecisionType PType>
class XPUMultiUpDecoderCompute : public KernelLite<TARGET(kXPU), PType> {
 public:
  using param_t = operators::XPUMultiUpDecoderParam;

  virtual void PrepareForRun();

  virtual void Run();

  virtual ~XPUMultiUpDecoderCompute() = default;

 private:
  xft::MultiUpDecoderParam multi_up_decoder_param_;

  std::vector<std::vector<std::vector<xft::xftVec<float>>>>
      xft_all_res_gn_weight_;
  std::vector<std::vector<std::vector<xft::xftVec<float>>>>
      xft_all_res_gn_bias_;
  std::vector<std::vector<std::vector<xft::xftVec<float>>>>
      xft_all_res_conv_bias_;
  std::vector<std::vector<std::vector<xft::xftTensor<int16_t, 4>>>>
      xft_all_res_conv_weight_;

  xft::xftVec<float> xft_last_gn_weight_;
  xft::xftVec<float> xft_last_gn_bias_;

  std::vector<std::vector<std::vector<const int16_t *>>>
      arg_all_res_conv_filter_int16_;
  std::vector<std::vector<std::vector<const float *>>> all_res_conv_filter_max_;
  std::vector<std::vector<std::vector<const float *>>> all_res_conv_input_max_;

  std::vector<xft::xftVec<float>> xft_all_post_conv_bias_;
  std::vector<xft::xftTensor<int16_t, 4>> xft_all_post_conv_weight_;
  std::vector<const int16_t *> arg_all_post_conv_filter_int16_;
  std::vector<const float *> all_post_conv_filter_max_;
  std::vector<const float *> all_post_conv_input_max_;

  XPUScratchPadGuard resblock_conv_filter_max_guard_;
  XPUScratchPadGuard post_conv_filter_max_guard_;

  template <typename T>
  std::vector<const T *> *GetPostConvFilter() {
    LOG(FATAL) << "Invalid Weight Type";
    return nullptr;
  }

  template <typename T>
  std::vector<std::vector<std::vector<const T *>>> *GetResConvFilter() {
    LOG(FATAL) << "Invalid Weight Type";
    return nullptr;
  }

  std::vector<const int16_t *> *GetPostConvFilter() {
    return &arg_all_post_conv_filter_int16_;
  }

  std::vector<std::vector<std::vector<const int16_t *>>> *GetResConvFilter() {
    return &arg_all_res_conv_filter_int16_;
  }

  void PreparePostConvFilterMax(const std::vector<lite::Tensor *> &weight_max,
                                int max_ptr_len,
                                std::vector<const float *> *max_xpu_ptrs);
  void PrepareAllResConvFilterMax(
      const std::vector<lite::Tensor *> &filter_max,
      const std::vector<std::vector<int>> &extra_info,
      int max_ptr_len,
      int conv_offset,
      std::vector<std::vector<std::vector<const float *>>> *max_xpu_ptrs);
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
