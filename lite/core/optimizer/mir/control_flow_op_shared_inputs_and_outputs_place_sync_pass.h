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

#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace mir {

// Sync the type of variable to the shared one in subblocks
//
// For example:
// graph[0]: main block
//                      in_x(target:x86)
//                       |
//                       |
//                       |
//                     while(target:host) ------- in_w(target:x86)
//                       |
//                       |
//                       |
//                     out_x(target:host)
//
// graph[1]: sub block
//                     in_x(target:xpu)
//                       |
//                       |
//                       |
//                      fc(target:xpu) ------ in_w(target:x86)
//                       |
//                       |
//                     softmax(target:xpu)
//                       |
//                       |
//                     out_x(target:xpu)
//
// After the pass is applied:
//
// graph[0]: main block
//                      in_x(target:x86)
//                       |
//                       |
//                       |
//                     while(target:host) ------- in_w(target:x86)
//                       |
//                       |
//                       |
//                     out_x(target:host)
//
// graph[1]: sub block
//                     in_x(target:x86)
//                       |
//                       |
//                       |
//                      fc(target:xpu) ------ in_w(target:x86)
//                       |
//                       |
//                     softmax(target:xpu)
//                       |
//                       |
//                     out_x(target:host)

class ControlFlowOpSharedInputsAndOutputsPlaceSyncPass : public mir::StmtPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph> &graph) override;
  void SetAllGraphs(std::vector<std::unique_ptr<mir::SSAGraph>> *graphs);

 private:
  std::vector<std::unique_ptr<mir::SSAGraph>> *graphs_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
