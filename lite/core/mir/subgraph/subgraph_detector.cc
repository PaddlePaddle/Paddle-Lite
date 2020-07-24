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

#include "lite/core/mir/subgraph/subgraph_detector.h"
#include <memory>
#include <set>
#include <utility>
#include <vector>
#include "lite/core/mir/dot.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher.h"
#include "lite/operators/subgraph_op.h"
#include "lite/utils/env.h"
#include "lite/utils/io.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {

std::string SubgraphVisualizer::operator()() {
  Dot dot;
  const std::vector<std::string> subgraph_colors{
      "red",          "green",          "cyan",           "bisque3",
      "coral",        "darkseagreen1",  "goldenrod1",     "darkorchid",
      "antiquewhite", "aquamarine",     "azure",          "bisque4",
      "blue2",        "brown1",         "burlywood1",     "cadetblue1",
      "chartreuse1",  "chocolate1",     "coral1",         "cornsilk",
      "crimson",      "cyan4",          "darkgoldenrod4", "darkolivegreen2",
      "darkorange2",  "darkorchid2",    "darkseagreen3",  "darkslategray",
      "deeppink2",    "deepskyblue2",   "dodgerblue",     "firebrick",
      "floralwhite",  "gold1",          "skyblue3",       "indianred",
      "indigo",       "lavenderblush2", "lightblue1",     "lightsalmon3",
      "khaki1",       "ivory4",         "sandybrown",     "olivedrab2",
      "turquoise4",   "snow3",          "sienna4",        "salmon2",
  };
  std::map<Node *, int> subgraph_indices;
  for (size_t i = 0; i < subgraphs_.size(); i++) {
    for (size_t j = 0; j < subgraphs_[i].size(); j++) {
      subgraph_indices[subgraphs_[i][j]] = i;
    }
  }
  std::map<std::string, int> exists_ops;
  std::set<std::string> exists_args;
  for (auto &node : graph_->StmtTopologicalOrder()) {
    if (!node->IsStmt()) {
      continue;
    }
    auto op_type = node->AsStmt().op_type();
    if (!exists_ops.count(op_type)) {
      exists_ops[op_type] = 0;
    } else {
      exists_ops[op_type]++;
    }
    auto op_name = op_type + paddle::lite::to_string(exists_ops[op_type]);
    std::string op_color = "white";
    if (subgraph_indices.count(node)) {
      auto subgraph_idx = subgraph_indices[node];
      op_name += "_subgraph_" + paddle::lite::to_string(subgraph_idx);
      op_color = subgraph_colors[subgraph_idx % subgraph_colors.size()];
    }
    dot.AddNode(op_name,
                {Dot::Attr("shape", "box"),
                 Dot::Attr("style", "filled"),
                 Dot::Attr("color", "black"),
                 Dot::Attr("fillcolor", op_color)});
    for (auto &in_node : node->inlinks) {
      auto arg_name = in_node->AsArg().name;
      if (!exists_args.count(arg_name)) {
        dot.AddNode(arg_name, {});
        exists_args.insert(arg_name);
      }
      dot.AddEdge(arg_name, op_name, {});
    }
    for (auto &out_node : node->outlinks) {
      auto arg_name = out_node->AsArg().name;
      if (!exists_args.count(arg_name)) {
        dot.AddNode(arg_name, {});
        exists_args.insert(arg_name);
      }
      dot.AddEdge(op_name, arg_name, {});
    }
  }

  auto res = dot.Build();
  std::cout << "subgraphs: " << subgraphs_.size() << "\n" << res << std::endl;
  return res;
}

// Find the ancestor node
SubgraphDetector::node_dat_t *
SubgraphDetector::node_dat_t::UnionFindAncestor() {
  node_dat_t *ancestor = this;
  while (ancestor->union_find_parent != ancestor) {
    ancestor = ancestor->union_find_parent;
  }
  return ancestor;
}

// Merge the two adjacent nodes into one node.
// Suppose we have two adjacent nodes src and dst.
// We will perform the following operations:
// 1. add all inputs(except src) of dst to src inlinks.
// 2. add all outputs of dst to src outlinks.
// 3. change all the dst's inputs and outputs
// corresponding inlinks and outlinks to src node.
// 4. delete all dst's inlinks and outlinks.
void SubgraphDetector::node_dat_t::UnionFindCombine(node_dat_t *candidate) {
  // Make this two node share the same ancestor.
  union_find_parent = UnionFindAncestor();
  node_dat_t *candidate_ancestor = candidate->UnionFindAncestor();
  candidate_ancestor->union_find_parent = union_find_parent;
  candidate->union_find_parent = union_find_parent;

  // Obtain the input and output nodes for the combined one
  std::set<node_dat_t *> inputs(inlinks.begin(), inlinks.end());
  std::set<node_dat_t *> outputs(candidate->outlinks.begin(),
                                 candidate->outlinks.end());
  for (auto *out_node : outlinks) {
    if (out_node != candidate) {
      outputs.insert(out_node);
    }
  }
  for (auto *in_node : candidate->inlinks) {
    if (in_node != this) {
      inputs.insert(in_node);
    }
  }

// Update the dst and src node's inlinks and outlinks.
#ifdef __clang__
  inlinks = node_set_t(inputs.begin(), inputs.end());
  outlinks = node_set_t(outputs.begin(), outputs.end());
  candidate->inlinks.clear();
  candidate->outlinks.clear();
#else
  inlinks = std::move(node_set_t(inputs.begin(), inputs.end()));
  outlinks = std::move(node_set_t(outputs.begin(), outputs.end()));
  candidate->inlinks.clear();
  candidate->outlinks.clear();
#endif

  // Change all the dst inputs and outputs corresponding inlink and
  // outlink to the src node.
  for (auto *in_node : inlinks) {
    for (auto *&out_node : in_node->outlinks) {
      if (out_node == candidate) {
        out_node = this;
      }
    }
  }
  for (auto *out_node : outlinks) {
    for (auto *&in_node : out_node->inlinks) {
      if (in_node == candidate) {
        in_node = this;
      }
    }
  }
}

// FlexibleDFS
// If reverse is true, do reverse dfs.
// If enter func is not nullptr, calls enter(node) before visiting any children
// of node.
// If leave func not nullptr, calls leave(node) after visiting all parents of
// node.
void SubgraphDetector::FlexibleDFS(
    const node_set_t &source,
    bool reverse,
    const std::function<bool(const node_dat_t *)> &enter,
    const std::function<bool(const node_dat_t *)> &leave) {
  std::vector<std::pair<const node_dat_t *, bool>> stack;  // node, leave
  for (auto &node : source) {
    stack.push_back(std::pair<const node_dat_t *, bool>(node, false));
  }
  std::set<const node_dat_t *> visited;
  while (!stack.empty()) {
    auto top = stack.back();
    stack.pop_back();

    if (top.second) {
      if (leave && !leave(top.first)) return;
    }
    if (visited.count(top.first)) continue;
    visited.insert(top.first);

    if (enter && !enter(top.first)) return;

    if (leave)
      stack.push_back(std::pair<const node_dat_t *, bool>(top.first, true));
    const node_set_t iter_nodes =
        reverse == true ? top.first->inlinks : top.first->outlinks;
    for (auto *node : iter_nodes) {
      if (!visited.count(node)) {
        stack.push_back(std::pair<const node_dat_t *, bool>(node, false));
      }
    }
  }
}

std::set<Node *> SubgraphDetector::GetExcludedNodesFromConfigFile() {
  // get exclude nodes from config file
  std::set<Node *> excluded_nodes;
  std::string config_file_path =
      GetStringFromEnv(SUBGRAPH_CUSTOM_PARTITION_CONFIG_FILE);
  if (!IsFileExists(config_file_path)) {
    return excluded_nodes;
  }
  std::vector<std::string> lines = ReadLines(config_file_path);

  for (std::string line : lines) {
    if (line.empty()) continue;
    std::vector<std::string> node_info = Split(line, ":");
    std::string op_type = node_info.at(0);
    std::vector<std::string> in_vars_name;
    if (node_info.size() > 1) {
      in_vars_name = Split(node_info.at(1), ",");
    }
    std::vector<std::string> out_vars_name;
    if (node_info.size() > 2) {
      out_vars_name = Split(node_info.at(2), ",");
    }

    for (auto &node : graph_->mutable_nodes()) {
      if (node.IsArg()) continue;
      auto stmt = node.stmt();
      if (op_type != stmt->op_type()) continue;
      auto in_nodes = node.inlinks;
      auto out_nodes = node.outlinks;
      if (in_vars_name.size() > in_nodes.size() ||
          out_vars_name.size() > out_nodes.size()) {
        continue;
      }

      bool matched = true;

      for (auto in_var_name : in_vars_name) {
        bool find_var = false;
        for (auto *in_node : in_nodes) {
          if (in_node->arg()->name == in_var_name) {
            find_var = true;
            break;
          }
        }
        if (!find_var) {
          matched = false;
          break;
        }
      }

      for (auto out_var_name : out_vars_name) {
        bool find_var = false;
        for (auto *out_node : out_nodes) {
          if (out_node->arg()->name == out_var_name) {
            find_var = true;
            break;
          }
        }
        if (!find_var) {
          matched = false;
          break;
        }
      }

      if (matched) {
        excluded_nodes.insert(&node);
      }
    }
  }

  return excluded_nodes;
}

void SubgraphDetector::InitNodes(node_map_t *nodes) {
  // Initialize and mark the subgraph detector nodes based on teller.
  std::set<Node *> excluded_nodes = GetExcludedNodesFromConfigFile();
  for (auto &it : *nodes) {
    for (auto &in_node : it.first->inlinks) {
      it.second->inlinks.push_back((*nodes)[in_node]);
    }
    for (auto &out_node : it.first->outlinks) {
      it.second->outlinks.push_back((*nodes)[out_node]);
    }
    if (teller_(it.first) && excluded_nodes.count(it.first) == 0) {
      it.second->marked = true;
      if (it.first->IsStmt()) {
        // If a function is inside the subgraph, mark all the output variables
        // to be inside too, so that two marked functions will be inside a same
        // subgraph, lets take a example:  A_function->var->B_function, if
        // A_function is marked, var should also be marked, so that B_function
        // will be in the same subgraph with A_function if B_function is
        // marked.
        for (auto &out_node : it.first->outlinks) {
          (*nodes)[out_node]->marked = true;
        }
      }
    }
  }
}  // namespace mir

std::vector<std::vector<Node *>> SubgraphDetector::ExtractSubgraphs(
    node_map_t *nodes) {
  for (auto &ordered_node : graph_->NodeTopologicalOrder()) {
    // different orders when traversing nodes in graph may lead to
    // different subgraph division, which may generate different result
    // with device such as MLU. These different results are all "right"
    // but a little confusing. Thus the topological order is used instead
    // of the address of the node in graph.
    CHECK(nodes->find(ordered_node) != nodes->end());
    node_dat_t *node = (*nodes)[ordered_node];
    if (!node->marked) {
      continue;
    }
    //  Our algorithm must guarantee that:
    //  1. The graph is always directed acyclic graph（DAG）.
    //  2. If there is a path in the subgraph from X to Y (X and Y are both
    //  nodes in the subgraph), then all paths from X to Y are in the
    //  subgraph.
    //
    //  In order to achieve the above guarantee.
    //  For adjacent nodes src -> dst.
    //  1. Get all dst input nodes except src.
    //  2. Reverse DFS from those input nodes
    //  3. If there is a path from input nodes to src,
    //  then the src and dst nodes can not be fused into one node,
    //  otherwise it can be done.
    while (true) {
      std::set<node_dat_t *> contract_nodes;
      for (auto *out_node : node->outlinks) {
        // must be an candidate
        if (!out_node->marked) continue;
        // get all dst input nodes except src node.
        node_set_t source_nodes;
        for (auto *in_node : out_node->inlinks) {
          if (in_node != node) {
            source_nodes.push_back(in_node);
          }
        }

        // Reverse DFS from the source_nodes.
        bool have_excess_path = false;
        FlexibleDFS(source_nodes,
                    true,
                    nullptr,
                    [&have_excess_path, node](const node_dat_t *n) {
                      if (n == node) {
                        have_excess_path = true;
                        return false;
                      }
                      return true;
                    });
        if (have_excess_path) continue;
        contract_nodes.insert(out_node);
      }
      if (contract_nodes.empty()) break;

      for (auto &contract_node : contract_nodes) {
        node->UnionFindCombine(contract_node);
      }
    }
  }

  std::map<node_dat_t * /*ancestor*/, std::vector<Node *>> clusters;
  for (auto &node : graph_->StmtTopologicalOrder()) {
    if (!node->IsStmt()) continue;
    if ((*nodes)[node]->marked) {
      clusters[(*nodes)[node]->UnionFindAncestor()].push_back(node);
    }
  }
  std::vector<std::vector<Node *>> subgraphs;
  std::for_each(clusters.begin(),
                clusters.end(),
                [&](const decltype(clusters)::value_type &it) {
                  subgraphs.push_back(it.second);
                });
  return subgraphs;
}

std::vector<std::vector<Node *>> SubgraphDetector::operator()() {
  node_map_t nodes;
  for (auto &node : graph_->mutable_nodes()) {
    nodes[&node] = new node_dat_t(&node);
    CHECK(nodes[&node]);
  }
  // Initialize and mark the subgraph detector nodes based on teller.
  InitNodes(&nodes);
  // Run the Extract algorithm to find all subgraphs.
  std::vector<std::vector<Node *>> subgraphs = ExtractSubgraphs(&nodes);
  for (auto &it : nodes) {
    CHECK(it.second);
    delete it.second;
  }
  return subgraphs;
}

void SubgraphFuser::InsertNewNode(SSAGraph *graph,
                                  int subgraph_idx,
                                  const std::vector<Node *> &subgraph_nodes) {
  // Create and attach a new subgraph op
  cpp::OpDesc subgraph_op_desc;
  subgraph_op_desc.SetType("subgraph");

  // Create a program desc and a block desc for storing all of Ops and Vars of
  // the target subgraph and sub_block_idx is set as a attribute of subgraph op,
  // sub_block_idx = 0 means it's a new subgraph op
  auto sub_program_desc = std::make_shared<cpp::ProgramDesc>();
  int sub_block_idx = 0;
  auto sub_block_desc = sub_program_desc->AddBlock<cpp::BlockDesc>();
  sub_block_desc->ClearOps();
  sub_block_desc->ClearVars();
  for (auto &op_node : subgraph_nodes) {
    auto sub_op_desc = sub_block_desc->AddOp<cpp::OpDesc>();
    *sub_op_desc = *op_node->AsStmt().op_info();
  }
  subgraph_op_desc.SetAttr<int32_t>("sub_block", sub_block_idx);

  // Extract input and output nodes from the target subgraph
  std::set<Node *> idata_var_nodes;
  std::set<Node *> weight_var_nodes;
  std::set<Node *> odata_var_nodes;
  std::set<Node *> local_var_nodes;
  std::set<Node *> unused_var_nodes;
  ExtractInputsOutputs(subgraph_nodes,
                       &idata_var_nodes,
                       &weight_var_nodes,
                       &odata_var_nodes,
                       &local_var_nodes,
                       &unused_var_nodes);
  // A simplified model without the original weight/local/unused nodes on the
  // subgraph ops will be saved only if 'SUBGRAPH_ONLINE_MODE' is set to
  // true(default) and Predictor->Run(...), Predictor->Save(...) is called.
  std::set<Node *> input_var_nodes(idata_var_nodes.begin(),
                                   idata_var_nodes.end());
  std::set<Node *> output_var_nodes(odata_var_nodes.begin(),
                                    odata_var_nodes.end());
  if (GetBoolFromEnv(SUBGRAPH_ONLINE_MODE, true)) {
    input_var_nodes.insert(weight_var_nodes.begin(), weight_var_nodes.end());
    output_var_nodes.insert(local_var_nodes.begin(), local_var_nodes.end());
    output_var_nodes.insert(unused_var_nodes.begin(), unused_var_nodes.end());
  }
  // Set input and output name mapping which stores the real inputs and
  // outputs
  std::vector<std::string> idata_var_names;
  std::vector<std::string> odata_var_names;
  for (auto &var_node : idata_var_nodes) {
    idata_var_names.push_back(var_node->AsArg().name);
  }
  for (auto &var_node : odata_var_nodes) {
    odata_var_names.push_back(var_node->AsArg().name);
  }
  subgraph_op_desc.SetAttr<std::vector<std::string>>("input_data_names",
                                                     idata_var_names);
  subgraph_op_desc.SetAttr<std::vector<std::string>>("output_data_names",
                                                     odata_var_names);
  // Set all of the inputs and outputs to the target subgraph op
  // To prevent vars are removed in RuntimeProgram::UpdateVarsOfProgram()
  std::vector<std::string> input_var_names;
  std::vector<std::string> output_var_names;
  for (auto &var_node : input_var_nodes) {
    input_var_names.push_back(var_node->AsArg().name);
  }
  for (auto &var_node : output_var_nodes) {
    output_var_names.push_back(var_node->AsArg().name);
  }
  subgraph_op_desc.SetInput("Inputs", input_var_names);
  subgraph_op_desc.SetOutput("Outputs", output_var_names);
  auto subgraph_op = LiteOpRegistry::Global().Create("subgraph");
  static_cast<operators::SubgraphOp *>(subgraph_op.get())
      ->SetProgramDesc(sub_program_desc);
  auto any_op = (*subgraph_nodes.begin())->AsStmt().op();
  subgraph_op->Attach(subgraph_op_desc, any_op->scope());

  // Export the scale values of the input/output var nodes of the inner op nodes
  // only for type_precision_cast_pass.
  for (auto &var_node : input_var_nodes) {
    auto var_node_name = var_node->arg()->name;
    auto any_op_node = var_node->outlinks.front();
    CHECK(any_op_node->IsStmt());
    auto &any_inst = any_op_node->AsStmt();
    if (any_inst.op_info()->HasInputScale(var_node_name)) {
      subgraph_op->mutable_op_info()->SetInputScale(
          var_node_name, any_inst.op_info()->GetInputScale(var_node_name));
    }
  }
  for (auto &var_node : output_var_nodes) {
    auto var_node_name = var_node->arg()->name;
    auto any_op_node = var_node->inlinks.front();
    CHECK(any_op_node->IsStmt());
    auto &any_inst = any_op_node->AsStmt();
    if (any_inst.op_info()->HasOutputScale(var_node_name)) {
      subgraph_op->mutable_op_info()->SetOutputScale(
          var_node_name, any_inst.op_info()->GetOutputScale(var_node_name));
    }
  }

  // Create and add a new subgraph node into the graph
  auto subgraph_op_node =
      graph->GraphCreateInstructNode(subgraph_op, any_op->valid_places());
  for (auto &var_node : input_var_nodes) {
    IR_NODE_LINK_TO(var_node, subgraph_op_node);
  }
  for (auto &var_node : output_var_nodes) {
    IR_OP_VAR_LINK(subgraph_op_node, var_node);
  }

  // Remove subgraph nodes and unused var nodes
  auto nodes2rm =
      GetNodes2RM(subgraph_nodes, {input_var_nodes, output_var_nodes});
  GraphSafeRemoveNodes(graph, nodes2rm);
}

void SubgraphFuser::ReplaceNodesWithSubgraphs(SSAGraph *graph,
                                              const SubgraphTeller &teller,
                                              int min_subgraph_size) {
  std::vector<std::vector<Node *>> subgraphs =
      SubgraphDetector(graph, teller)();
  SubgraphVisualizer(graph, subgraphs)();
  for (size_t subgraph_idx = 0; subgraph_idx < subgraphs.size();
       subgraph_idx++) {
    if (subgraphs[subgraph_idx].size() >= min_subgraph_size) {
      InsertNewNode(graph, subgraph_idx, subgraphs[subgraph_idx]);
    }
  }
}

void SubgraphFuser::operator()() {
  ReplaceNodesWithSubgraphs(graph_, teller_, min_subgraph_size_);
}

void ExtractInputsOutputs(const std::vector<Node *> &op_nodes,
                          std::set<Node *> *input_var_nodes,
                          std::set<Node *> *weight_var_nodes,
                          std::set<Node *> *output_var_nodes,
                          std::set<Node *> *local_var_nodes,
                          std::set<Node *> *unused_var_nodes) {
  for (auto &op_node : op_nodes) {
    for (auto &var_node : op_node->inlinks) {
      if (var_node->AsArg().is_weight) {
        weight_var_nodes->insert(var_node);
        continue;
      }
      if (!var_node->inlinks.empty()) {
        // Var can only come from one op node, so use front
        auto *prev_op_node = var_node->inlinks.front();
        if (std::find(op_nodes.begin(), op_nodes.end(), prev_op_node) !=
            op_nodes.end()) {
          continue;
        }
      }
      input_var_nodes->insert(var_node);
    }
    for (auto &var_node : op_node->outlinks) {
      if (var_node->outlinks.empty()) {
        // The next op is empty so this var is actually unused
        unused_var_nodes->insert(var_node);
        continue;
      }
      // Var can have more than one next op node, So, if all next nodes are in
      // op_nodes then it should be put into local_var_nodes
      bool next_op_in_nodes = true;
      for (auto &next_op_node : var_node->outlinks) {
        if (std::find(op_nodes.begin(), op_nodes.end(), next_op_node) ==
            op_nodes.end()) {
          next_op_in_nodes = false;
          break;
        }
      }
      if (next_op_in_nodes) {
        local_var_nodes->insert(var_node);
        continue;
      }
      output_var_nodes->insert(var_node);
    }
  }
}

std::set<const Node *> GetNodes2RM(
    const std::vector<Node *> &op_nodes,
    const std::vector<std::set<Node *>> &excluded_var_nodes) {
  std::set<const Node *> nodes2rm(op_nodes.begin(), op_nodes.end());
  for (auto &op_node : op_nodes) {
    for (auto &var_node : op_node->inlinks) {
      bool skip = false;
      // skip the var node which is used by any other ops that doesn't belong to
      // the subgraph ops.
      for (auto &out_op_node : var_node->outlinks) {
        if (std::find(op_nodes.begin(), op_nodes.end(), out_op_node) !=
            op_nodes.end()) {
          skip = true;
          break;
        }
      }
      if (!skip && !nodes2rm.count(var_node)) {
        nodes2rm.insert(var_node);
      }
    }
    for (auto &var_node : op_node->outlinks) {
      if (!nodes2rm.count(var_node)) {
        nodes2rm.insert(var_node);
      }
    }
  }
  // Excluded nodes should not be removed
  for (auto &excluded_var_node : excluded_var_nodes) {
    for (auto &var_node : excluded_var_node) {
      if (nodes2rm.count(var_node)) {
        nodes2rm.erase(var_node);
      }
    }
  }
  return nodes2rm;
}

static void SortHelper(Node *node,
                       const std::set<Node *> &unordered_nodes,
                       std::set<const Node *> *visited_nodes,
                       std::vector<Node *> *ordered_nodes) {
  for (auto &var_node : node->inlinks) {
    if (var_node->inlinks.empty()) continue;
    auto *op_node = var_node->inlinks.front();
    if (unordered_nodes.count(op_node) && !visited_nodes->count(op_node)) {
      SortHelper(op_node, unordered_nodes, visited_nodes, ordered_nodes);
    }
  }
  ordered_nodes->push_back(node);
  visited_nodes->insert(node);
}

std::vector<Node *> GetTopologicalOrder(
    const std::set<Node *> &unordered_nodes) {
  std::set<const Node *> visited_nodes;
  std::vector<Node *> ordered_nodes;
  for (auto &node : unordered_nodes) {
    if (!node->IsStmt()) continue;
    if (visited_nodes.count(node)) continue;
    SortHelper(node, unordered_nodes, &visited_nodes, &ordered_nodes);
  }
  return ordered_nodes;
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle
