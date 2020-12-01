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

#include <cmath>
#include <string>
#include <vector>
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <PrecisionType Ptype, PrecisionType Otype>
class GemmLikeConv : public KernelLite<TARGET(kARM), Ptype> {
 public:
  GemmLikeConv() = default;
  ~GemmLikeConv() {}

  virtual void ReInitWhenNeeded();
  virtual void PrepareForRun();
  virtual void Run();

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
  }

  std::string kernel_func_name_{"NotImplForConvGemm"};
#endif

  /// todo, support inplace weights transform
 protected:
  using param_t = operators::ConvParam;
  DDim last_shape_;
  std::vector<float> w_scale_;
  bool flag_1x1gemm_{true};
  bool flag_trans_weights_{false};
  bool flag_trans_bias_{false};
  Tensor weights_;
  Tensor bias_;
  int workspace_size_{0};
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
