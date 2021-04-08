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

// In SSD, the concat after prior-box and reshape ops, the prior-box and reshape
// can be calculate offline in "priorbox eliminate pass" and "reshape eliminate
// pass"
// so the concat can be calculate offline, too. we support concat which link 6
// reshape
// at present.
//
// For example:
// reshape-output     reshape-output     ..other reshape-output..
//       |                    |                     |
//       |                    |                     |
//       |                    |                     |
//       |                    |                     |
//       -------------- OP: concat ---------------
//                            |
//                            |
//                            |
//                            |
//                            v
//
// After the pass is applied:
//                      concat-output
//                            |
//                            |
//                            |
//                            |
//                            v

class ConcatEliminator : public FuseBase {
 public:
  void BuildPattern() override;
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override;

 protected:
  void DeleteInterNodes(SSAGraph* graph) override;

 private:
  void ComputeConcat(const std::vector<lite::Tensor*> inputs,
                     lite::Tensor* output);
  std::vector<size_t> StrideNumel(const DDim& ddim);
  std::set<const Node*> nodes2rm_;
};

class ConcatEliminatePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
