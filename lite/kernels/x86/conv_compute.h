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

#include <Eigen/Core>
#include <string>
#include <vector>
#include "lite/backends/x86/math/avx/conv_utils.h"
#include "lite/backends/x86/math/blas.h"
#include "lite/backends/x86/math/conv_bias.h"
#include "lite/backends/x86/math/gemm_s8u8_compute.h"
#include "lite/backends/x86/math/im2col.h"
#include "lite/backends/x86/math/vol2col.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"
#include "lite/operators/conv_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace x86 {

inline bool IsExpand(const std::vector<int64_t>& filter_dim,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations) {
  bool filter_1 = true, strides_1 = true, padding_0 = true, dilation_1 = true;
  for (size_t j = 0; j < strides.size(); ++j) {
    filter_1 = filter_1 && (static_cast<int>(filter_dim[j + 2]) == 1);
    strides_1 = strides_1 && (strides[j] == 1);
    padding_0 = padding_0 && (paddings[j] == 0);
    dilation_1 = dilation_1 && (dilations[j] == 1);
  }
  return !(filter_1 && strides_1 && padding_0 && dilation_1);
}

template <PrecisionType Ptype, PrecisionType OutType>
class Conv2dCompute : public KernelLite<TARGET(kX86), Ptype> {
 public:
  virtual void PrepareForRun();

  virtual void ReInitWhenNeeded() {
    if (impl_) {
      impl_->ReInitWhenNeeded();
    }
  }

  virtual void Run();

#ifdef LITE_WITH_PROFILE
  std::string kernel_func_name_{"Conv2d"};
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = "NotImplForConv";
  }
#endif

  ~Conv2dCompute() {
    if (impl_ != nullptr) {
      delete impl_;
    }
    for (int i = 0; i < gemm_s8_ptr_float_.size(); i++)
      delete gemm_s8_ptr_float_[i];
    for (int i = 0; i < gemm_s8_ptr_int8_.size(); i++)
      delete gemm_s8_ptr_int8_[i];
  }

 private:
  using param_t = operators::ConvParam;
  KernelLite<TARGET(kX86), Ptype>* impl_{nullptr};
  Context<TargetType::kX86>* device_ctx;
  bool flag_1x1gemm_{false};
  bool flag_trans_bias_{true};
  std::vector<float> w_scale_;
  Tensor weights_;
  Tensor bias_;
  std::vector<lite::x86::math::generate_gemm_s8u8_x86_kern<float>*>
      gemm_s8_ptr_float_{};
  std::vector<lite::x86::math::generate_gemm_s8u8_x86_kern<int8_t>*>
      gemm_s8_ptr_int8_{};
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
