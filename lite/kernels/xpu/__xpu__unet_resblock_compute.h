// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

template <typename T>
struct identity {
  typedef T type;
};

template <typename InType, PrecisionType PType>
class XPUSpatialTransformerResBlockCompute
    : public KernelLite<TARGET(kXPU), PType> {
 public:
  using param_t = operators::XPUSpatialTransformerResBlockParam;

  virtual void PrepareForRun();

  virtual void Run();

  virtual ~XPUSpatialTransformerResBlockCompute() = default;

 private:
  std::vector<const int16_t *> arg_fc_weight_int16_;
  std::vector<const int16_t *> arg_conv_filter_int16_;
  std::vector<const float *> arg_fc_bias_;
  std::vector<const float *> arg_conv_bias_;
  std::vector<const float *> arg_gn_scale_;
  std::vector<const float *> arg_gn_bias_;
  std::vector<const float *> fc_weight_max_;
  std::vector<const float *> conv_filter_max_;
  std::vector<const float *> input_max_;
  XPUScratchPadGuard weight_max_guard_;
  XPUScratchPadGuard filter_max_guard_;

  template <typename T>
  std::vector<const T *> *get_weight() {
    return get_weight_identity(identity<T>());
  }
  template <typename T>
  std::vector<const T *> *get_weight_identity(identity<T>) {
    LOG(FATAL) << "Invalid Weight Type";
    return nullptr;
  }

  std::vector<const int16_t *> *get_weight_identity(identity<int16_t>) {
    return &arg_fc_weight_int16_;
  }

  template <typename T>
  std::vector<const T *> *get_filter() {
    return get_filter_identity(identity<T>());
  }
  template <typename T>
  std::vector<const T *> *get_filter_identity(identity<T>) {
    LOG(FATAL) << "Invalid Weight Type";
    return nullptr;
  }

  std::vector<const int16_t *> *get_filter_identity(identity<int16_t>) {
    return &arg_conv_filter_int16_;
  }

  void prepare_weight_max(const std::vector<lite::Tensor *> &weight_max,
                          int max_ptr_len,
                          std::vector<const float *> *max_xpu_ptrs);
  void prepare_filter_max(const std::vector<lite::Tensor *> &filter_max,
                          int max_ptr_len,
                          std::vector<const float *> *max_xpu_ptrs);
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
