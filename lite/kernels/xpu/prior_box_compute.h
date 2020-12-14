// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

class PriorBoxCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::PriorBoxParam;

  void PrepareForRun() override;

  virtual void Run();

  virtual ~PriorBoxCompute() = default;

 private:
  XPUScratchPadGuard xpu_aspect_ratios_guard_;
  XPUScratchPadGuard xpu_min_sizes_guard_;
  XPUScratchPadGuard xpu_max_sizes_guard_;
  XPUScratchPadGuard variance_xpu_guard_;
  int prior_num;
  int ar_num;
  int min_size_num;
  int max_size_num;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
