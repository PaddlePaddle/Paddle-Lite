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
class ConvCompute : public KernelLite<TARGET(kARM), Ptype> {
 public:
  virtual void PrepareForRun();

  virtual void ReInitWhenNeeded() {
    CHECK(impl_);
    impl_->ReInitWhenNeeded();
  }

  virtual void Run() {
    CHECK(impl_);
    impl_->Run();
  }

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    impl_->SetProfileRuntimeKernelInfo(ch);
  }
#endif

  ~ConvCompute() {
    if (impl_ != nullptr) {
      delete impl_;
    }
  }

 private:
  using param_t = operators::ConvParam;
  KernelLite<TARGET(kARM), Ptype>* impl_{nullptr};
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
