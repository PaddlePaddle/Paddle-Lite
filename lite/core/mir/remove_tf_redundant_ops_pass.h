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
#include "lite/core/mir/pass.h"
#include "lite/core/tensor.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * mir::RemoveTFRedundantOpsPass remove reshape2->squeeze2 pattern
 * and last reshape2 op for tensorflow mobilenetv1/v2.
 */
class RemoveTFRedundantOpsPass : public mir::StmtPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
  void RemoveReshape2Pattern(const std::unique_ptr<SSAGraph>& graph);
  void RemoveSqueeze2Reshape2Pattern(const std::unique_ptr<SSAGraph>& graph);
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
