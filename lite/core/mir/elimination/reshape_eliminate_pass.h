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
#include <set>
#include <string>
#include <vector>
#include "lite/core/mir/pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {

// In SSD, the reshape ops after prior-box, the prior-box can be calculate
// offline
// in "priorbox eliminate pass", so the reshape can be calculate offline, too
//
// For example:
//     boxes                             variances
//       |                                   |
//       |                                   |
//       |                                   |
//       |                                   |
//   OP: reshape                         OP: reshape
//       |                                   |
//       v                                   v
//
// After the pass is applied:
// reshape's output                    reshape's output
//       |                                   |
//       |                                   |
//       |                                   |
//       |                                   |
//       v                                   v

class ReshapeEliminator : public FuseBase {
 public:
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 protected:
  void DeleteInterNodes(SSAGraph* graph) override;

 private:
  void ComputeReshape(const lite::Tensor* in, lite::Tensor* out);
  std::set<const Node*> nodes2rm_;
};

class ReshapeEliminatePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
