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

  // Below function cloud be useful in child classes //
  // classify node by subgraph id
  std::unordered_map<int, std::unordered_set<Node*>> ClassifySubgraph(
      const std::unique_ptr<SSAGraph>& graph);

  // generate the graph op desc
  cpp::OpDesc GenGraphOpDesc(const std::string& model_name,
                             const std::vector<std::string>& in_var_names,
                             const std::vector<std::string>& out_var_names);

  // insert a new graph op node
  void InsertNewNode(const std::unique_ptr<SSAGraph>& graph,
                     const std::string& model_name,
                     Scope* scope,
                     const std::vector<Place>& valid_places,
                     std::unordered_set<Node*> in_data_vars,
                     std::unordered_set<Node*> in_wgt_vars,
                     std::unordered_set<Node*> out_data_vars,
                     std::unordered_set<Node*> out_unused_vars);

  // Sort and return the topology order of nodes set
  std::vector<Node*> GetTopologicalOrder(
      const std::unordered_set<Node*>& nodes);

  // find all input data vars, input weight vars,
  // output data vars and output vars from the nodes
  void FindInputOutputVars(const std::unordered_set<Node*>& op_nodes,
                           std::unordered_set<Node*>* in_data_vars,
                           std::unordered_set<Node*>* in_wgt_vars,
                           std::unordered_set<Node*>* out_data_vars,
                           std::unordered_set<Node*>* out_unused_vars);

  // return the node to remove in the subgraph
  std::unordered_set<const Node*> GetNode2rm(
      const std::unordered_set<Node*>& op_nodes,
      const std::vector<std::unordered_set<Node*>>& excluded_nodes);

 private:
  // sort nodes to operational sequence
  void SortHelper(Node* node,
                  const std::unordered_set<Node*>& nodes_all,
                  std::unordered_set<const Node*>* visited_nodes,
                  std::vector<Node*>* ret);
};

}  // namespace subgraph
}  // namespace mir
}  // namespace lite
}  // namespace paddle
