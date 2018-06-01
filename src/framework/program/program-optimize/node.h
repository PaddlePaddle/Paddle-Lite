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

#include <map>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "common/log.h"
#include "framework/paddle_mobile_object.h"
#include "framework/program/op_desc.h"

namespace paddle_mobile {
namespace framework {

class Node : PaddleMobileObject {
  friend class ProgramOptimize;

 public:
  Node() {}
  explicit Node(const std::string &type) : type_(type) {}
  explicit Node(std::shared_ptr<OpDesc> op_desc)
      : op_desc_(op_desc), type_(op_desc->Type()) {}
  Node &operator>(std::shared_ptr<Node> node);
  bool operator==(const Node &in);
  bool CanSplit(std::unordered_set<std::string> complex_compute_set);
  std::string ToString() const;
  std::shared_ptr<Node> To(int size);
  uint Depth(uint begin = 0);
  Node &Folder(
      uint size, std::string type,
      std::map<std::string, std::pair<std::string, std::string>> change_map);
  std::vector<std::shared_ptr<framework::OpDesc>> OpDescs(uint size);
  std::vector<std::shared_ptr<framework::OpDesc>> OpDescs();
  std::shared_ptr<framework::OpDesc> OpDescOfNode() { return op_desc_; }
  std::string Type() { return type_; }
  void Description();

 private:
  void CanSplit(bool *split, bool spliting, int complex_count,
                std::unordered_set<std::string> *complex_compute_set,
                Node *pre_node);
  void OpDescs(std::vector<std::shared_ptr<framework::OpDesc>> *op_desc,
               Node *node, bool adding_thread, int thread_num);
  void OpDescs(uint size,
               std::vector<std::shared_ptr<framework::OpDesc>> *op_desc);
  void To(int index, std::shared_ptr<Node>);
  void Folder(
      std::shared_ptr<framework::OpDesc> op_desc,
      std::vector<std::shared_ptr<Node>> *outputs, uint index,
      std::map<std::string, std::pair<std::string, std::string>> *change,
      Node *begin_node);
  std::shared_ptr<framework::OpDesc> op_desc_;
  std::string ToString(std::string blank, const Node *node) const;
  std::vector<std::shared_ptr<Node>> outputs_;
  std::vector<Node *> inputs_;
  std::string type_;
};

Print &operator<<(Print &printer, const Node &node);
}  // namespace framework
}  // namespace paddle_mobile
