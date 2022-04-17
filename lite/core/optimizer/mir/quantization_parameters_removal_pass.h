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
#include <set>
#include <string>
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * Clear ops' quant information by config file
 *
 * for ops to clear quant info
 * before:
 *   in_var(int8) -> op(int8) -> out_var(int8)
 * after:
 *   in_var(int8) -> op -> out_var
 */
class QuantizationParametersRemovalPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

 private:
  std::string GetMixedPrecisionQuantizationConfig(Scope* scope);
  std::set<Node*> GetTargetNodesFromMixedPrecisionQuantizationConfig(
      const std::unique_ptr<SSAGraph>& graph,
      const std::string& mixed_precision_quantization_config);
  void ClearQuantInfo(paddle::lite::mir::Node* node);
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
