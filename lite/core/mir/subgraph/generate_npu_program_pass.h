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
#include <unordered_set>
#include <vector>
#include "lite/core/mir/pass.h"
#include "lite/core/mir/subgraph/subgraph_program_pass.h"
#include "lite/npu/bridge/registry.h"
#include "lite/npu/npu_helper.h"

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
  // TODO(TJ): maybe change a name
  // convert all fused subgraphs to npu clients
  // 1. if some subgraph failed, then skip.
  // 2. add new graph nodes, kernels and context
  // 3. remove unused nodes
  void ConvertSubgraph(const std::unique_ptr<SSAGraph>& graph, int sub_num);

  // call convert function from start node
  // return if convert success and the nodes to remove
  // return the output(arg.name, npu op)
  lite::npu::bridge::node_map_type CvtOpNodes(
      const lite::npu::bridge::cvt_map_type& cvtfunc_map,
      const Node* op_node,
      const lite::npu::bridge::node_map_type& inputs_map,
      int sub_id,
      std::unordered_set<const Node*>* nodes2rm,
      key2nodes_t* matched);

 private:
  std::vector<Instruction> insts_;
};

}  // namespace subgraph
}  // namespace mir
}  // namespace lite
}  // namespace paddle
