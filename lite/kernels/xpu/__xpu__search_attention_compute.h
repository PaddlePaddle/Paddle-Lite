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
#include "lite/backends/xpu/target_wrapper.h"  // XPUScratchPadGuard
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

class XPUMmdnnSearchAttentionCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::XPUMmdnnSearchAttentionParam;

  void PrepareForRun() override;

  void Run() override;

 private:
  XPUScratchPadGuard offset_xpu_guard_;
  XPUScratchPadGuard pad_begin_xpu_guard_;
  XPUScratchPadGuard w_max_xpu_guard_;
  XPUScratchPadGuard buffer_at_l3_guard_;
  XPUScratchPadGuard buffer_at_gm_guard_;

  std::unique_ptr<int[]> offset_cpu;
  std::unique_ptr<int[]> pad_begin_cpu;

  const int L3_SLOT_SIZE = 40 * 128 * 128;
  const int GM_SLOT_SIZE = 40 * 512 * 512;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
