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
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * Only for nnadapter: insert calib op to support mixed precision model.
 *
 * PS: for cpu, calib op will be insert by type_precision_cast_pass.
 */
class NNAdapterInsertCalibPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

 private:
  void ComplementInputs(const std::unique_ptr<SSAGraph>& graph,
                        Node* in,
                        Node* inst_node,
                        std::map<std::string, Node*>* cast_nodes);
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
