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

#include <unordered_set>
#include <utility>
#include <vector>
#include "core/types.h"

namespace nnadapter {

class ModelPartitioner {
 public:
  // This is a simple representation of a model. It holds the pointer of the
  // Node to avoid changing the graph during the graph partition.
  struct Node {
    explicit Node(core::Operation *o) : operation(o) {}
    core::Operation *operation;
    int class_id{-1};
    Node *union_find_parent{this};
    std::vector<Node *> inlinks{};
    std::vector<Node *> outlinks{};
    Node *UnionFindAncestor();
    void UnionFindCombine(Node *candidate);
  };

  ModelPartitioner() {}
  ModelPartitioner(const ModelPartitioner &) = delete;
  ModelPartitioner(ModelPartitioner &&) = default;
  ModelPartitioner &operator=(const ModelPartitioner &) = delete;
  virtual ~ModelPartitioner() {}

  void Apply(
      core::Model *model,
      const std::vector<std::pair<int, std::unordered_set<core::Operation *>>>
          &supported_operations,
      std::vector<std::pair<int, std::vector<core::Operation *>>> *subgraphs);

 private:
  void FlexibleDFS(const std::vector<Node *> &sources,
                   bool reverse,
                   const std::function<bool(const Node *)> &enter,
                   const std::function<bool(const Node *)> &leave);
};

void PartitionModelIntoSubmodels(
    core::Model *model,
    const std::vector<std::pair<int, std::unordered_set<core::Operation *>>>
        &supported_operations,
    std::vector<std::pair<int, core::Model *>> *models,
    std::vector<std::vector<int>> *input_indexes,
    std::vector<std::vector<int>> *output_indexes);

}  // namespace nnadapter
