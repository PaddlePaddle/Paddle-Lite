// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include "lite/core/kernel.h"
#include "lite/operators/generate_proposals_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

class GenerateProposalsCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::GenerateProposalsParam;

  void Run() override;

  void PrepareForRun() override;

  virtual ~GenerateProposalsCompute() = default;

 private:
  XPUScratchPadGuard trans_scores_guard_;
  XPUScratchPadGuard trans_deltas_guard_;
  XPUScratchPadGuard im_info_guard_;
  XPUScratchPadGuard box_sel_guard_;
  XPUScratchPadGuard scores_sel_guard_;
  XPUScratchPadGuard index_sel_guard_;
  XPUScratchPadGuard num_guard_;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
