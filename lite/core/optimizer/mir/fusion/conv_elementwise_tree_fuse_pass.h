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

#include <memory>
#include <string>
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

// This pass fuse conv2d_1x1 and elementwise_add.
// As elementwise_add is low ratio of computation / memory_access. This fuse
// will reduce the access to output of conv2d_1x1 and then increase the whole
// low ratio of computation / memory_access.
//
// For example:
//
//       |                          |
//       |                          |
//       A                     conv2d_1x1
//       |                          |
//       |                          |
//       ----- elementwise_add -----
//                  |
//                  |
//                  V
//
// After the pass is applied:
//
//       |                          |
//       |                          |
//       A                          |
//       |                          |
//       |                          |
//       -------- conv2d_1x1 -------
//                    |
//                    |
//                    V
//
// Limitations:
// * Only support fuse of conv2d_1x1 and elementwise_add and
//   fusion_elementwise_add_activation.
// * The input tensor dims of elementwise_add/fusion_elementwise_add_activation
//   must be equal to that of conv2d_1x1.
// * The output tensor of conv2d_1x1 must be Y of
//   elementwise_add/fusion_elementwise_add_activation.
// * Only support opencl target by now.

class ConvElementwiseTreeFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
