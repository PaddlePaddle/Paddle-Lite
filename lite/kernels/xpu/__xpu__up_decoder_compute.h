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
#include "layers/spatial_transformer.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

namespace xft = baidu::xpu::xft;

template <typename InType, PrecisionType PType>
class XPUUpDecoderCompute : public KernelLite<TARGET(kXPU), PType> {
 public:
  using param_t = operators::XPUUpDecoderParam;

  virtual void PrepareForRun();

  virtual void Run();

  virtual ~XPUUpDecoderCompute() = default;

 private:
  std::vector<xft::xftVec<float>> xft_resblock_gn_weight_;
  std::vector<xft::xftVec<float>> xft_resblock_gn_bias_;
  std::vector<xft::xftVec<float>> xft_resblock_conv_bias_;
  std::vector<xft::xftTensor<int16_t, 4>> xft_resblock_conv_weights_;

  std::vector<const int16_t *> arg_resblock_conv_filter_int16_;
  std::vector<const float *> resblock_conv_filter_max_;
  std::vector<const float *> resblock_input_max_;
  XPUScratchPadGuard resblock_conv_filter_max_guard_;
  XPUScratchPadGuard post_conv_filter_max_guard_;

  template <typename T>
  std::vector<const T *> *GetWeight() {
    LOG(FATAL) << "Invalid Weight Type";
    return nullptr;
  }

  template <typename T>
  std::vector<const T *> *GetFilter() {
    LOG(FATAL) << "Invalid Weight Type";
    return nullptr;
  }

  std::vector<const int16_t *> *GetFilter() {
    return &arg_resblock_conv_filter_int16_;
  }

  void PreparePostConvFilterMax(const std::vector<lite::Tensor *> &weight_max,
                                int max_ptr_len,
                                std::vector<const float *> *max_xpu_ptrs);
  void PrepareResblockConvFilterMax(
      const std::vector<lite::Tensor *> &filter_max,
      int max_ptr_len,
      std::vector<const float *> *max_xpu_ptrs);
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
