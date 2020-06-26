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

class SearchGrnnCompute : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::SearchGrnnParam;

  void PrepareForRun() override;

  void prepare_layout(const operators::SearchGrnnParam& param,
                      const paddle::lite::Tensor* bottom);
  void Run() override;

 private:
  XPUScratchPadGuard offset_xpu_guard_;
  XPUScratchPadGuard new_offset_xpu_guard_;
  XPUScratchPadGuard maxs_xpu_guard_;

  std::unique_ptr<int[]> idx_sorted_by_width_data_cpu;
  std::unique_ptr<int[]> offset_cpu;
  std::unique_ptr<int[]> new_offset_cpu;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
