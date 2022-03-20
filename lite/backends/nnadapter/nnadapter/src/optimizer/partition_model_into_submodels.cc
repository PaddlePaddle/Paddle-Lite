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

#include "optimizer/partition_model_into_submodels.h"
#include <functional>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/utility.h"

namespace nnadapter {

// Find the ancestor node
ModelPartitioner::Node *ModelPartitioner::Node::UnionFindAncestor() {
  Node *ancestor = this;
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
void ModelPartitioner::Node::UnionFindCombine(Node *candidate) {
  // Make this two node share the same ancestor.
  union_find_parent = UnionFindAncestor();
  auto candidate_ancestor = candidate->UnionFindAncestor();
  candidate_ancestor->union_find_parent = union_find_parent;
  candidate->union_find_parent = union_find_parent;
  // Obtain the input and output nodes for the combined one
  std::set<Node *> input_nodes(inlinks.begin(), inlinks.end());
  std::set<Node *> output_nodes(candidate->outlinks.begin(),
                                candidate->outlinks.end());
  for (auto output_node : outlinks) {
    if (output_node != candidate) {
      output_nodes.insert(output_node);
    }
  }
  for (auto input_node : candidate->inlinks) {
    if (input_node != this) {
      input_nodes.insert(input_node);
    }
  }
// Update the dst and src node's inlinks and outlinks.
#ifdef __clang__
  inlinks = std::vector<Node *>(input_nodes.begin(), input_nodes.end());
  outlinks = std::vector<Node *>(output_nodes.begin(), output_nodes.end());
#else
  inlinks =
      std::move(std::vector<Node *>(input_nodes.begin(), input_nodes.end()));
  outlinks =
      std::move(std::vector<Node *>(output_nodes.begin(), output_nodes.end()));
#endif
  candidate->inlinks.clear();
  candidate->outlinks.clear();
  // Change all the dst inputs and outputs corresponding inlink and
  // outlink to the src node.
  for (auto input_node : inlinks) {
    for (auto &output_node : input_node->outlinks) {
      if (output_node == candidate) {
        output_node = this;
      }
    }
  }
  for (auto output_node : outlinks) {
    for (auto &input_node : output_node->inlinks) {
      if (input_node == candidate) {
        input_node = this;
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
void ModelPartitioner::FlexibleDFS(
    const std::vector<Node *> &sources,
    bool reverse,
    const std::function<bool(const Node *)> &enter,
    const std::function<bool(const Node *)> &leave) {
  std::vector<std::pair<const Node *, bool>> stack;  // Node, leave
  for (auto source : sources) {
    stack.push_back(std::pair<const Node *, bool>(source, false));
  }
  std::set<const Node *> visited;
  while (!stack.empty()) {
    auto top = stack.back();
    stack.pop_back();
    if (top.second) {
      if (leave && !leave(top.first)) return;
    }
    if (visited.count(top.first)) continue;
    visited.insert(top.first);
    if (enter && !enter(top.first)) return;
    if (leave) stack.push_back(std::pair<const Node *, bool>(top.first, true));
    const std::vector<Node *> neighbors =
        reverse ? top.first->inlinks : top.first->outlinks;
    for (auto neighbor : neighbors) {
      if (!visited.count(neighbor)) {
        stack.push_back(std::pair<const Node *, bool>(neighbor, false));
      }
    }
  }
}

void ModelPartitioner::Apply(
    core::Model *model,
    const std::vector<std::pair<int, std::unordered_set<core::Operation *>>>
        &supported_operations,
    std::vector<std::pair<int, std::vector<core::Operation *>>> *subgraphs) {
  // Create the nodes to represent the origin model and mark the supported
  // operations
  std::vector<std::shared_ptr<Node>> nodes;
  std::map<core::Operation *, Node *> operation_to_node_map;
  auto find_or_add_node = [&](core::Operation *operation) {
    NNADAPTER_CHECK(operation);
    if (!operation_to_node_map.count(operation)) {
      auto node = std::make_shared<Node>(operation);
      NNADAPTER_CHECK(node)
          << "Failed to allocate node for the model partition, out of memory!";
      nodes.push_back(node);
      for (auto &_supported_operations_ : supported_operations) {
        if (_supported_operations_.second.count(operation)) {
          node->class_id = _supported_operations_.first;
          break;
        }
      }
      operation_to_node_map[operation] = node.get();
    }
    return operation_to_node_map[operation];
  };
  auto operations = SortOperationsInTopologicalOrder(model);
  for (auto operation : operations) {
    auto node = find_or_add_node(operation);
    for (auto input_operand : operation->input_operands) {
      auto prev_operation = GetOperandProducer(model, input_operand);
      if (prev_operation) {
        node->inlinks.push_back(find_or_add_node(prev_operation));
      }
    }
    for (auto output_operand : operation->output_operands) {
      auto next_operations = GetOperandConsumers(model, output_operand);
      for (auto next_operation : next_operations) {
        node->outlinks.push_back(find_or_add_node(next_operation));
      }
    }
  }
  // Run to extract all subgraphs
  for (auto operation : operations) {
    // Different orders when traversing nodes in graph may lead to
    // different subgraph division, which may generate different result
    // with device such as MLU. These different results are all "right"
    // but a little confusing. Thus the topological order is used instead
    // of the address of the node in graph.
    NNADAPTER_CHECK(operation_to_node_map.count(operation));
    auto node = operation_to_node_map[operation];
    if (node->class_id < 0) continue;
    //  Our algorithm must guarantee that:
    //  1. The graph is always directed acyclic graph（DAG）.
    //  2. If there is a path in the subgraph from X to Y (X and Y are both
    //  nodes in the subgraph), then all paths from X to Y are in the
    //  subgraph.
    //
    //  In order to achieve the above guarantee.
    //  For adjacent nodes between src_node -> dst_node.
    //  1. Get the input nodes of the dst_node except the src_node.
    //  2. Reverse DFS from those input nodes.
    //  3. If there is only one path from the input nodes to the src_node, the
    //  src_node and the dst_node are clustered into the same cluster and share
    //  the same ancestor.
    while (true) {
      std::set<Node *> contract_nodes;
      for (auto output_node : node->outlinks) {
        // Must be an candidate
        if (output_node->class_id < 0 ||
            output_node->class_id != node->class_id)
          continue;
        // Get the input nodes of the dst_node except the src_node.
        std::vector<Node *> source_nodes;
        for (auto input_node : output_node->inlinks) {
          if (input_node != node) {
            source_nodes.push_back(input_node);
          }
        }
        // Reverse DFS from the source_nodes.
        bool have_excess_path = false;
        FlexibleDFS(source_nodes,
                    true,
                    nullptr,
                    [&have_excess_path, node](const Node *_node_) {
                      if (_node_ == node) {
                        have_excess_path = true;
                        return false;
                      }
                      return true;
                    });
        if (have_excess_path) continue;
        contract_nodes.insert(output_node);
      }
      if (contract_nodes.empty()) break;
      for (auto contract_node : contract_nodes) {
        node->UnionFindCombine(contract_node);
      }
    }
  }
  Node *ancestor = nullptr;
  std::unordered_map<Node *, int> indexes;
  subgraphs->clear();
  for (auto operation : operations) {
    int index = -1;
    auto node = operation_to_node_map[operation];
    if (node->class_id >= 0) {
      ancestor = node->UnionFindAncestor();
      if (!indexes.count(ancestor)) {
        indexes[ancestor] = subgraphs->size();
        subgraphs->emplace_back(node->class_id,
                                std::vector<core::Operation *>());
      }
      index = indexes[ancestor];
      ancestor = nullptr;
    } else {
      if (!ancestor) {
        ancestor = node;
        indexes[ancestor] = subgraphs->size();
        subgraphs->emplace_back(-1, std::vector<core::Operation *>());
      }
      index = indexes[ancestor];
    }
    subgraphs->at(index).second.push_back(operation);
  }
  auto subgraph_count = subgraphs->size();
  NNADAPTER_VLOG(5) << subgraph_count << " subgraphs detected!";
  for (size_t i = 0; i < subgraph_count; i++) {
    auto node_count = subgraphs->at(i).second.size();
    NNADAPTER_VLOG(5) << "#" << i << " subgraph has " << node_count
                      << " nodes for class_id=#" << subgraphs->at(i).first;
    for (size_t j = 0; j < node_count; j++) {
      NNADAPTER_VLOG(5) << "op " << OperationTypeToString(
                                        subgraphs->at(i).second.at(j)->type);
    }
  }
}

NNADAPTER_EXPORT void PartitionModelIntoSubmodels(
    core::Model *model,
    const std::vector<std::pair<int, std::unordered_set<core::Operation *>>>
        &supported_operations,
    std::vector<std::pair<
        int,
        std::tuple<core::Model *, bool, std::vector<int>, std::vector<int>>>>
        *models) {
  // Partition the model into the subgraphs
  ModelPartitioner partitioner;
  std::vector<std::pair<int, std::vector<core::Operation *>>> subgraphs;
  partitioner.Apply(model, supported_operations, &subgraphs);
  // Create the submodels from the subgraphs
  models->clear();
  // Mapping a shared operand to share index
  std::map<core::Operand *, int> shared_operand_to_shared_index_map;
  for (auto &subgraph : subgraphs) {
    auto _model_ = new core::Model();
    NNADAPTER_CHECK(_model_)
        << "Failed to allocate for a model, out of memory!";
    auto class_id = subgraph.first;
    std::vector<int> input_indexes, output_indexes;
    // Mapping an old operand to an new operand
    std::unordered_map<core::Operand *, core::Operand *>
        old_operand_to_new_operand_map;
    auto &old_operations = subgraph.second;
    for (auto old_operation : old_operations) {
      auto new_operation = AddOperation(_model_);
      *new_operation = *old_operation;
      for (auto &old_operand : new_operation->input_operands) {
        if (!old_operand) continue;
        core::Operand *new_operand = nullptr;
        if (old_operand_to_new_operand_map.count(old_operand)) {
          new_operand = old_operand_to_new_operand_map[old_operand];
        } else {
          new_operand = AddOperand(_model_);
          // The buffer of operand should not be freed when the operand is
          // deleted.
          CopyOperand(new_operand, old_operand, false);
          old_operand_to_new_operand_map[old_operand] = new_operand;
          if (IsModelInputOperand(new_operand)) {
            _model_->input_operands.push_back(new_operand);
            input_indexes.push_back(
                -GetModelInputOperandIndex(model, old_operand) - 1);
          } else if (!IsConstantOperand(new_operand)) {
            auto prev_operation = GetOperandProducer(model, old_operand);
            if (prev_operation &&
                std::find(old_operations.begin(),
                          old_operations.end(),
                          prev_operation) == old_operations.end()) {
              if (!shared_operand_to_shared_index_map.count(old_operand)) {
                shared_operand_to_shared_index_map[old_operand] =
                    shared_operand_to_shared_index_map.size();
              }
              new_operand->type.lifetime = NNADAPTER_MODEL_INPUT;
              _model_->input_operands.push_back(new_operand);
              input_indexes.push_back(
                  shared_operand_to_shared_index_map[old_operand]);
            }
          }
        }
        old_operand = new_operand;
      }
      for (auto &old_operand : new_operation->output_operands) {
        if (!old_operand) continue;
        core::Operand *new_operand = nullptr;
        if (old_operand_to_new_operand_map.count(old_operand)) {
          new_operand = old_operand_to_new_operand_map[old_operand];
        } else {
          new_operand = AddOperand(_model_);
          // The buffer of operand should not be freed when the operand is
          // deleted.
          CopyOperand(new_operand, old_operand, false);
          old_operand_to_new_operand_map[old_operand] = new_operand;
          if (IsModelOutputOperand(new_operand)) {
            _model_->output_operands.push_back(new_operand);
            output_indexes.push_back(
                -GetModelOutputOperandIndex(model, old_operand) - 1);
          } else if (!IsConstantOperand(new_operand)) {
            bool all_in_subgraph = true;
            auto next_operations = GetOperandConsumers(model, old_operand);
            for (auto next_operation : next_operations) {
              if (std::find(old_operations.begin(),
                            old_operations.end(),
                            next_operation) == old_operations.end()) {
                all_in_subgraph = false;
                break;
              }
            }
            if (!all_in_subgraph) {
              if (!shared_operand_to_shared_index_map.count(old_operand)) {
                shared_operand_to_shared_index_map[old_operand] =
                    shared_operand_to_shared_index_map.size();
              }
              new_operand->type.lifetime = NNADAPTER_MODEL_OUTPUT;
              _model_->output_operands.push_back(new_operand);
              output_indexes.push_back(
                  shared_operand_to_shared_index_map[old_operand]);
            }
          }
        }
        old_operand = new_operand;
      }
    }
    NNADAPTER_VLOG(6) << "#" << models->size() << " submodel for class_id=#"
                      << class_id << std::endl
                      << Visualize(model);
    models->emplace_back(
        class_id,
        std::make_tuple(_model_, false, input_indexes, output_indexes));
  }
  NNADAPTER_VLOG(5) << models->size() << " submodels detected!";
}

}  // namespace nnadapter
