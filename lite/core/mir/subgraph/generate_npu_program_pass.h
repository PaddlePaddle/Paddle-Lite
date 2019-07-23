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
#include "lite/npu/bridge/registry.h"
#include "lite/npu/npu_helper.h"

namespace paddle {
namespace lite {
namespace mir {

class GenerateNPUProgramPass : public ProgramPass {
 public:
  using key2nodes_t = std::map<std::string, Node*>;

  void Apply(const std::unique_ptr<SSAGraph>& graph) override;
  std::unique_ptr<RuntimeProgram> GenProgram();

 protected:
  void InferOnce(const std::unique_ptr<SSAGraph>& graph);

  // clear all subgraph id and mark all ops, which could be fuse, as id zero
  void InitSubgraphID(const std::unique_ptr<SSAGraph>& graph);

  // TODO(TJ): maybe change a name
  // convert all fused subgraphs to npu clients
  // 1. if some subgraph failed, then skip.
  // 2. add new graph nodes, kernels and context
  // 3. remove unused nodes
  void ConvertSubgraph(const std::unique_ptr<SSAGraph>& graph, int sub_num);

  // // GenerateFusedGraph:
  // std::unique_ptr<SSAGraph> GenerateFusedGraph(const
  // std::unique_ptr<SSAGraph>& graph, int sub_num);
  void ChangeAllOutConnectedID(Node* node, int to_id, int from_id = 0);

  // make all the linked ops in subgraph with same subgraph_id
  // return the fused subgraph numbers
  int FuseSubgraphID(const std::unique_ptr<SSAGraph>& graph);

  // call convert function from start node
  // return if convert success and the nodes to remove
  // return the output npu op
  std::shared_ptr<ge::Operator> CvtOpNodes(
      const npu::bridge::Factory::map_type& cvtfunc_map,
      const Node* op_node,
      std::shared_ptr<ge::Operator> input,
      int sub_id,
      std::unordered_set<const Node*>* nodes2rm,
      key2nodes_t* matched);

  // change current node and all connected output node id from id to another id
  void ChangeOutConnectedID(Node* node, int to_id, int from_id = 0);

 private:
  std::vector<Instruction> insts_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
