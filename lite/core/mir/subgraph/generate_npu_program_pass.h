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

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/npu_helper.h"
#include "lite/core/mir/pass.h"
#include "lite/core/mir/subgraph/subgraph_program_pass.h"

namespace paddle {
namespace lite {
namespace mir {
namespace subgraph {

class GenerateNPUProgramPass : public SubgraphProgramPass {
 public:
  using key2nodes_t = std::map<std::string, Node*>;

  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
  std::unique_ptr<RuntimeProgram> GenProgram();

 protected:
  // nodes2cvt: op nodes to convert
  // return cvted_vars: converted var nodes
  void CvtAllOpNodes(const std::vector<Node*>& nodes2cvt,
                     lite::npu::bridge::node_map_type* cvted_vars);

  std::shared_ptr<ge::Operator> CvtVarNode(lite::mir::Node* var_node,
                                           const Scope* scope);

  std::string BuildNPUGraph(const std::unordered_set<Node*>& op_nodes,
                            const std::unordered_set<Node*>& in_data_vars,
                            const std::unordered_set<Node*>& out_data_vars,
                            int sub_id);

  void GenNPUSubgraph(const std::unique_ptr<SSAGraph>& graph,
                      const std::unordered_set<Node*>& op_nodes,
                      int sub_id);

 private:
  std::vector<Instruction> insts_;
};

}  // namespace subgraph
}  // namespace mir
}  // namespace lite
}  // namespace paddle
