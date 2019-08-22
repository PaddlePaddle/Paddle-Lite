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
#include "lite/core/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {
namespace subgraph {

class SubgraphProgramPass : public ProgramPass {
 public:
  using key2nodes_t = std::map<std::string, Node*>;

  // make all the linked ops in subgraph with same subgraph_id
  // return the fused subgraph numbers
  int FuseSubgraph(const std::unique_ptr<SSAGraph>& graph,
                   const std::vector<std::string>& supported_op_types);

  void Apply(const std::unique_ptr<SSAGraph>& graph) override{};

 protected:
  void InferOnce(const std::unique_ptr<SSAGraph>& graph);

  // clear all subgraph id and mark all ops, which could be fuse, as id zero
  void InitSubgraphID(const std::unique_ptr<SSAGraph>& graph,
                      const std::vector<std::string>& supported_op_types);

  // make all the linked ops in subgraph with same subgraph_id
  // return the fused subgraph numbers
  int FuseSubgraphID(const std::unique_ptr<SSAGraph>& graph);

  // // GenerateFusedGraph:
  // std::unique_ptr<SSAGraph> GenerateFusedGraph(const
  // std::unique_ptr<SSAGraph>& graph, int sub_num);
  void ChangeAllOutConnectedID(Node* node, int to_id, int from_id = 0);

 private:
  // {1: {nodes2rm_in_subgraph1, ...},
  //  2: {nodes2rm_in_subgraph2, ...}}
  // delete nodes
  std::unordered_map<int, std::unordered_set<Node*>> nodes2rm_;
  // std::unordered_map<int, std::unordered_set<const Node*>> nodes2rm_;
  // inputs nodes
  std::unordered_map<int, std::unordered_set<Node*>> i_nodes_;
  // std::unordered_map<int, std::unordered_set<const Node*>> i_nodes_;
  // outputs nodes
  std::unordered_map<int, std::unordered_set<Node*>> o_nodes_;
  // std::unordered_map<int, std::unordered_set<const Node*>> o_nodes_;
};

}  // namespace subgraph
}  // namespace mir
}  // namespace lite
}  // namespace paddle
