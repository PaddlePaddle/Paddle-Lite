/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cinttypes>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "common/log.h"
#include "framework/program/op_desc.h"

namespace paddle_mobile {
namespace framework {

class Node {
  friend class ProgramOptimize;

 public:
  Node() {}
  explicit Node(const std::string &type) : type_(type) {}
  explicit Node(std::shared_ptr<OpDesc> op_desc)
      : op_desc_(op_desc), type_(op_desc->Type()) {}
  Node &operator>(std::shared_ptr<Node> node);
  bool operator==(const Node &in);
  bool MedianEqual(const Node &in);

#ifdef PADDLE_MOBILE_DEBUG
  std::string ToString() const;
  void Description();
#endif
  std::shared_ptr<Node> To(int size);
  int Depth(int begin = 0);
  Node &Folder(
      int size, std::string type,
      std::map<std::string, std::vector<std::pair<std::string, std::string>>>
          change,
      std::vector<std::shared_ptr<Node>> *removed_nodes);
  std::shared_ptr<framework::OpDesc> OpDescOfNode() { return op_desc_; }
  std::string Type() { return type_; }

  std::vector<Node *> operator[](int index);

  std::map<std::string, Node *> Relationship();

 private:
  void RelationshipPrivate(std::map<std::string, Node *> *map);
  void GetNodesWithLocation(int index, int now_index,
                            std::vector<Node *> *nodes);
  void To(int index, std::shared_ptr<Node>);
  void Folder(
      std::shared_ptr<framework::OpDesc> op_desc,
      std::vector<std::shared_ptr<Node>> *outputs, int index,
      std::map<std::string, std::vector<std::pair<std::string, std::string>>>
          *change,
      Node *begin_node, std::vector<std::shared_ptr<Node>> *removed_nodes);
  std::shared_ptr<framework::OpDesc> op_desc_;
#ifdef PADDLE_MOBILE_DEBUG
  std::string ToString(std::string blank, const Node *node) const;
#endif
  std::vector<std::shared_ptr<Node>> outputs_;
  std::vector<Node *> inputs_;
  std::string type_;
};

Print &operator<<(Print &printer, const Node &node);
}  // namespace framework
}  // namespace paddle_mobile
