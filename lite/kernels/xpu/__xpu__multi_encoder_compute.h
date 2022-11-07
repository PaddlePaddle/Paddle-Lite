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

#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

class XPUMultiEncoderCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::XPUMultiEncoderParam;

  virtual void PrepareForRun();

  virtual void Run();

 private:
  std::vector<const int8_t *> arg_fc_weight_int8_;
  std::vector<const int16_t *> arg_fc_weight_int16_;
  std::vector<const float *> arg_fc_weight_fp32_;
  std::vector<const float16 *> arg_fc_weight_fp16_;
  std::vector<const float *> arg_fc_bias_;
  std::vector<const float *> arg_ln_scale_;
  std::vector<const float *> arg_ln_bias_;
  std::vector<const float *> fc_weight_max_;
  std::vector<const float *> fc_input_max_;
  std::vector<const float *> roformer_embedding_;
  std::vector<xdnn::QuantType> quant_types_;
  XPUScratchPadGuard weight_max_guard_;
  XPUScratchPadGuard input_max_guard_;
  XPUScratchPadGuard cast_in_guard_;
  XPUScratchPadGuard cast_out_guard_;
  xdnn::Activation_t qkv_act = xdnn::Activation_t::RELU;
  int slice_idx = -1;
  int relative_type_ = 0;
  bool local_quant_ = false;

  template <typename T>
  std::vector<const T *> *get_weight();

  void prepare_quant_max(const std::vector<float> &max_value,
                         int n_layers,
                         int max_ptr_len,
                         std::vector<const float *> &max_xpu_ptrs);
  void prepare_weight_max(bool per_channel,
                          const std::vector<lite::Tensor *> &weight_max,
                          int max_ptr_len,
                          std::vector<const float *> &max_xpu_ptrs);
  template <typename T, typename TW, typename TGEMM>
  void run_encoder(const T *in, T *out);
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
