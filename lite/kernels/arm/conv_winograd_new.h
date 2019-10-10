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
#include "lite/backends/arm/math/conv_impl.h"
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

/// only support 3x3s1 and 3x3s2
template <PrecisionType Ptype, PrecisionType OutType>
class WinogradConv : public KernelLite<TARGET(kARM), Ptype> {
 public:
  WinogradConv() = default;
  ~WinogradConv() {}
  virtual void PrepareForRun();
  virtual void ReInitWhenNeeded();
  virtual void Run();

 protected:
  using param_t = operators::ConvParam;
  Tensor weights_;
  DDim last_shape_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
