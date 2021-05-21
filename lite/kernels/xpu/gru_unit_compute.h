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
// WIfloatHOUfloat WARRANfloatIES OR CONDIfloatIONS OF ANY KIND, either express
// or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

class GRUUnitCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::GRUUnitParam;

  void PrepareForRun() override;

  virtual void Run();

  virtual ~GRUUnitCompute() = default;

 private:
  float weight_s1_abs_max_;
  float weight_s2_abs_max_;
  XPUScratchPadGuard weight_max_guard_;
  XPUScratchPadGuard quant_weight_guard_;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
