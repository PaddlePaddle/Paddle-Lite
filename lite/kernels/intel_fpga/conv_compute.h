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
#include <memory>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace intel_fpga {

template <PrecisionType Ptype, PrecisionType OutType>
class ConvCompute : public KernelLite<TARGET(kIntelFPGA), Ptype> {
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

  ~ConvCompute() {
    if (impl_ != nullptr) {
      delete impl_;
    }
  }

 private:
  using param_t = operators::ConvParam;
  std::unique_ptr<KernelContext> arm_cxt_{nullptr};
  KernelLite<TARGET(kARM), Ptype>* impl_{nullptr};
};

}  // namespace intel_fpga
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
