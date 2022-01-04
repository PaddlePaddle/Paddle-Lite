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
#include <set>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

using SubgraphTeller = std::function<bool(Node*)>;

class SubgraphVisualizer {
 public:
  SubgraphVisualizer(SSAGraph* graph,
                     const std::vector<std::vector<Node*>>& subgraphs)
      : graph_(graph), subgraphs_(subgraphs) {}
  std::string operator()();

 protected:
  SSAGraph* graph_{nullptr};
  std::vector<std::vector<Node*>> subgraphs_;
};

/*
 * Divide the graph into subgraphs according to the specified conditions.
 * Return the divided clusters, a cluster is consisted of the op nodes in the
 * subgraph.
 */
class SubgraphDetector {
 public:
  // This is a simple representation of a graph. The SDNode hold the
  // pointer of the Node. This is to avoid changing the original graph in the
  // process of graph analysis.
  struct node_dat_t;
  using node_map_t = std::map<Node*, node_dat_t*>;
  using node_set_t = std::vector<node_dat_t*>;
  struct node_dat_t {
    explicit node_dat_t(Node* _node) : node(_node) {}
    Node* node;
    bool marked{false};
    node_dat_t* union_find_parent{this};
    node_set_t inlinks{};
    node_set_t outlinks{};
    node_dat_t* UnionFindAncestor();
    void UnionFindCombine(node_dat_t* candidate);
  };

  SubgraphDetector(SSAGraph* graph,
                   const SubgraphTeller& teller,
                   const std::string& subgraph_partition_configs = "")
      : graph_(graph),
        teller_(teller),
        subgraph_partition_configs_(subgraph_partition_configs) {}
  std::vector<std::vector<Node*>> operator()();

  void FlexibleDFS(const node_set_t& source,
                   bool reverse,
                   const std::function<bool(const node_dat_t*)>& enter,
                   const std::function<bool(const node_dat_t*)>& leave);

  std::set<Node*> GetExcludedNodesFromSubgraphPartitionConfigs(
      const std::string& subgraph_partition_configs);

  void InitNodes(node_map_t* nodes);

  std::vector<std::vector<Node*>> ExtractSubgraphs(node_map_t* nodes);

 protected:
  SSAGraph* graph_{nullptr};
  SubgraphTeller teller_;
  const std::string& subgraph_partition_configs_;
};

/*
 * Replace all of subgraphs with the subgraph ops, a block desc is added into
 * the subgraph op to wrap the original op nodes, keep all of var nodes of the
 * original ops nodes as the inputs and outputs of the subgraph op
 */
class SubgraphFuser {
 public:
  SubgraphFuser(SSAGraph* graph,
                const SubgraphTeller& teller,
                int min_subgraph_size,
                const std::string& subgraph_partition_configs = "",
                bool support_mixed_precision = false)
      : graph_(graph),
        teller_(teller),
        min_subgraph_size_{min_subgraph_size},
        subgraph_partition_configs_(subgraph_partition_configs),
        support_mixed_precision_(support_mixed_precision) {}
  void operator()();

  // Remove the op nodes of the subgraphs and replace with the subgraph ops.
  void ReplaceNodesWithSubgraphs(
      SSAGraph* graph,
      const std::vector<std::vector<Node*>>& subgraphs,
      int min_subgraph_size);
  // Create a subgraph node with a block desc to wrap the original op nodes of
  // the subgraph
  void InsertNewNode(SSAGraph* graph,
                     int subgraph_idx,
                     const std::vector<Node*>& subgraph_nodes);

 protected:
  SSAGraph* graph_{nullptr};
  SubgraphTeller teller_;
  int min_subgraph_size_;
  const std::string& subgraph_partition_configs_;
  bool support_mixed_precision_{false};
};

class MixedPrecisionAutoInsertCalibFuser {
 public:
  MixedPrecisionAutoInsertCalibFuser(SSAGraph* graph,
                                     std::vector<std::vector<Node*>>* subgraphs)
      : graph_(graph), subgraphs_(subgraphs) {}

  void operator()();

  // Quant ops' out precision should be int8
  void UpdateQuantOpOut(const std::vector<Node*>& nodes);

  // before:
  //    non_quant_op -> var0 -> quant_op
  // after:
  //    non_quant_op -> var0 -> calib -> var1 -> quant_op
  void InsertQuantCalib(SSAGraph* graph, std::vector<Node*>* nodes);

  // before:
  //    quant_op -> var0 -> non_quant_op
  // after:
  //    quant_op -> var0 -> calib -> var1 -> non_quant_op
  void InsertDeQuantCalib(SSAGraph* graph, std::vector<Node*>* nodes);

 private:
  SSAGraph* graph_{nullptr};
  std::vector<std::vector<Node*>>* subgraphs_{nullptr};
};

void ExtractInputsOutputs(const std::vector<Node*>& op_nodes,
                          std::set<Node*>* input_var_nodes,
                          std::set<Node*>* weight_var_nodes,
                          std::set<Node*>* output_var_nodes,
                          std::set<Node*>* local_var_nodes,
                          std::set<Node*>* unused_var_nodes);

std::set<const Node*> GetNodes2RM(
    const std::vector<Node*>& op_nodes,
    const std::vector<std::set<Node*>>& excluded_var_nodes);

std::vector<Node*> GetTopologicalOrder(const std::set<Node*>& unordered_nodes);

}  // namespace mir
}  // namespace lite
}  // namespace paddle
