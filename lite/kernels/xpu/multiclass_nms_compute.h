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
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

class MulticlassNmsCompute
    : public KernelLite<TARGET(kXPU), PRECISION(kFloat)> {
 public:
  using param_t = operators::MulticlassNmsParam;

  void Run() override;

  void PrepareForRun() override;

  virtual ~MulticlassNmsCompute() = default;

 private:
  XPUScratchPadGuard unnormalized_boxes_guard_;
  XPUScratchPadGuard unnormalize_box_offset_guard_;

  XPUScratchPadGuard topk_index_guard_;
  XPUScratchPadGuard topk_scores_guard_;
  XPUScratchPadGuard xpu_score_thres_guard_;
  XPUScratchPadGuard topk_scores_lower_bound_index_guard_;
  XPUScratchPadGuard topk_boxes_guard_;

  XPUScratchPadGuard nms_index_guard_;
  XPUScratchPadGuard nms_scores_guard_;
  XPUScratchPadGuard nms_class_guard_;
  XPUScratchPadGuard nms_keep_box_num_guard_;
  XPUScratchPadGuard nms_boxes_index_guard_;

  XPUScratchPadGuard batch_box_index_guard_;
  XPUScratchPadGuard merge_box_index_guard_;
  XPUScratchPadGuard merge_index_guard_;
  XPUScratchPadGuard merge_scores_guard_;
  XPUScratchPadGuard merge_boxes_guard_;
  XPUScratchPadGuard merge_class_guard_;
  XPUScratchPadGuard batch_offset_guard_;
};

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
