// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <list>
#include <map>
#include <memory>
#include <set>
#include <stack>
#include <string>
#include <vector>
#include "utility/debug.h"
#include "utility/logging.h"
#include "utility/micros.h"
#include "utility/modeling.h"
#include "utility/node.h"
#include "utility/string.h"
#include "utility/utility.h"

namespace nnadapter {

class Graph {
 public:
  Graph() = default;

  ~Graph();

  // @param model: the nnadapter model with operands and operations)
  void Build(core::Model *model);

  void RemoveNode(const Node *node);

  // Clone from another Graph, all Node(s) are duplicated.
  void CloneFrom(const Graph &from);

  std::vector<Node *> OperationTopologicalOrder();

  std::vector<Node *> NodeTopologicalOrder();

  // The inputs of the graph.
  std::vector<Node *> inputs();

  // The outputs of the graph.
  std::vector<Node *> outputs();

  const std::list<Node> &nodes() const { return node_storage_; }
  std::list<Node> &mutable_nodes() { return node_storage_; }

  Node *NewArgumentNode();
  Node *NewInstructNode();

  void CheckValid() {
    NNADAPTER_CHECK(CheckBidirectionalConnection());
    NNADAPTER_CHECK(CheckNodesRoleSet());
    NNADAPTER_CHECK(CheckLinksRoleSet());
  }

  Node *GraphCreateInstructNode(core::Operation &op);  // NOLINT

 private:
  // Check the bidirectional connection.
  bool CheckBidirectionalConnection();
  bool CheckNodesRoleSet();
  // Check all the items's role in inlinks and outlinks is set.
  bool CheckLinksRoleSet();

  // Build operator inlink edge table.
  std::map<Node *, std::set<Node *>> BuildOperationAdjList();

  // Build node inlink edge table.
  std::map<Node *, std::set<Node *>> BuildNodeAdjList();

  void SortHelper(const std::map<Node *, std::set<Node *>> &adj_list,
                  Node *node,
                  std::set<Node *> *visited,
                  std::vector<Node *> *ret);

 private:
  std::list<Node> node_storage_;
};

// Remove the link between a -> b.
static void RemoveDirectedLink(Node *a, Node *b) {
  auto it = std::find(b->inlinks.begin(), b->inlinks.end(), a);
  if (it != b->inlinks.end()) {
    b->inlinks.erase(it);
  }

  auto it1 = std::find(a->outlinks.begin(), a->outlinks.end(), b);
  if (it1 != a->outlinks.end()) {
    a->outlinks.erase((it1));
  }
}

// Link a -> b.
static void DirectedLink(Node *a, Node *b) {
  // Eagerly remove first, to avoid duplicate link.
  RemoveDirectedLink(a, b);
  a->outlinks.push_back(b);
  b->inlinks.push_back(a);
}

}  // namespace nnadapter
