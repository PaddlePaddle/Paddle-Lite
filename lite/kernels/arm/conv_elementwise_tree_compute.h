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
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/kernel.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <PrecisionType Ptype, PrecisionType OutType>
class ConvElementwiseTreeCompute : public KernelLite<TARGET(kARM), Ptype> {
 public:
  virtual void PrepareForRun();

  virtual void ReInitWhenNeeded() {
    CHECK(conv_impl_);
    conv_impl_->ReInitWhenNeeded();
    CHECK(elt_impl_);
    elt_impl_->ReInitWhenNeeded();
  }

  virtual void Run() {
    CHECK(conv_impl_);
    conv_impl_->Run();
    CHECK(elt_impl_);
    elt_impl_->Run();
  }

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    impl_->SetProfileRuntimeKernelInfo(ch);
  }
#endif

  ~ConvElementwiseTreeCompute() {
    if (conv_impl_ != nullptr) {
      delete conv_impl_;
    }
    if (elt_impl_ != nullptr) {
      delete elt_impl_;
    }
  }

 private:
  using param_t = operators::FusionConvElementParam;
  KernelLite<TARGET(kARM), Ptype>* conv_impl_{nullptr};
  KernelLite<TARGET(kARM), Ptype>* elt_impl_{nullptr};
  // temp tensor
  Tensor tmp_output_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
