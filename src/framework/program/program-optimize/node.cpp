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

#include <sstream>

#include "framework/program/program-optimize/node.h"

namespace paddle_mobile {

namespace framework {

Node &Node::operator>(std::shared_ptr<Node> node) {
  outputs_.push_back(node);
  std::shared_ptr<Node> this_node;
  node->inputs_.push_back(this);
  return *node;
}

bool Node::operator==(const Node &in) {
  if (in.type_ == this->type_) {
    if (this->outputs_.size() == in.outputs_.size()) {
      for (int i = 0; i < outputs_.size(); ++i) {
        if (!(*outputs_[i] == *in.outputs_[i])) {
          return false;
        }
      }
    } else {
      return false;
    }
  } else {
    return false;
  }
  return true;
}

// std::shared_ptr<Node> Node::MatchTheFirstNode(std::string type){
//
//  for (const auto &node : outputs_){
//    if (node->type_ == type){
//      return node;
//    }else{
//
//    }
//  }
//}

std::vector<std::shared_ptr<framework::OpDesc>> Node::OpDescs(uint size) {
  std::vector<std::shared_ptr<framework::OpDesc>> op_descs;
  OpDescs(size - 1, &op_descs);
  return op_descs;
}

void Node::OpDescs(uint index,
                   std::vector<std::shared_ptr<framework::OpDesc>> *op_desc) {
  if (index == 0) {
    return;
  }
  op_desc->push_back(this->op_desc_);
  for (auto &output : outputs_) {
    output->OpDescs(index, op_desc);
  }
}

void Node::OpDescs(std::vector<std::shared_ptr<framework::OpDesc>> *op_desc,
                   Node *node) {
  auto iter = std::find(op_desc->begin(), op_desc->end(), this->op_desc_);
  if (inputs_.size() > 1 && node != inputs_.back()) {
    return;
  } else if (inputs_.size() > 1 && node == inputs_.back()) {
    op_desc->push_back(this->op_desc_);
  } else {
    op_desc->push_back(this->op_desc_);
  }

  for (auto &output : outputs_) {
    output->OpDescs(op_desc, this);
  }
}

std::vector<std::shared_ptr<framework::OpDesc>> Node::OpDescs() {
  std::vector<std::shared_ptr<framework::OpDesc>> op_descs;
  OpDescs(&op_descs, this);
  return op_descs;
}

std::string Node::ToString(std::string blank, const Node *node) const {
  std::stringstream ss;
  ss << type_ << "-> \n";

  if (inputs_.size() > 1 && node != inputs_.back()) {
    return ss.str();
  } else if (inputs_.size() > 1 && node == inputs_.back()) {
    ss << "\n" << blank << type_ << "\n";
  }

  for (int i = 0; i < outputs_.size(); ++i) {
    ss << blank << outputs_[i]->ToString(blank + "  ", this) << "";
  }
  return ss.str();
}

std::string Node::ToString() const { return this->ToString("  ", this); }

std::shared_ptr<Node> Node::To(int size) {
  std::shared_ptr<Node> node = std::make_shared<Node>();
  this->To(size - 1, node);
  return node;
}

// Node &Node::To(int size) {
//  if (size == 1) {
//    this->outputs_.clear();
//  }
//
//  for (int j = 0; j < this->outputs_.size(); ++j) {
//    outputs_[j]->To(size - 1);
//  }
//  return *this;
//}

void Node::To(int index, std::shared_ptr<Node> node) {
  node->type_ = this->type_;
  if (index != 0) {
  } else {
    return;
  }

  for (int j = 0; j < this->outputs_.size(); ++j) {
    std::shared_ptr<Node> sub_node = std::make_shared<Node>();
    node->outputs_.push_back(sub_node);
    outputs_[j]->To(index - 1, sub_node);
  }
}

uint Node::Depth(uint begin) {
  uint depth = 0;
  begin++;
  for (int i = 0; i < outputs_.size(); ++i) {
    uint output_depth = outputs_[i]->Depth(begin);
    depth = output_depth > depth ? output_depth : depth;
  }
  return begin > depth ? begin : depth;
}

Node &Node::Folder(
    uint size, std::string type,
    std::map<std::string, std::pair<std::string, std::string>> change) {
  std::shared_ptr<framework::OpDesc> op_desc =
      std::make_shared<framework::OpDesc>();
  op_desc->inputs_ = this->op_desc_->inputs_;
  std::vector<std::shared_ptr<Node>> outputs;
  this->Folder(op_desc, &outputs, size - 1, &change, this);
  this->outputs_ = outputs;
  this->type_ = type;
  this->op_desc_ = op_desc;
  this->op_desc_->type_ = type;
  return *this;
}

void Node::Folder(
    std::shared_ptr<framework::OpDesc> op_desc,
    std::vector<std::shared_ptr<Node>> *outputs, uint index,
    std::map<std::string, std::pair<std::string, std::string>> *change,
    Node *begin_node) {
  if (change->find(this->type_) != change->end()) {
    auto change_pair = (*change)[this->type_];
    op_desc->GetInputs()[change_pair.second] =
        this->op_desc_->GetInputs()[change_pair.first];
  }

  for (auto &attr_pair : this->op_desc_->attrs_) {
    op_desc->attrs_.emplace(attr_pair.first, attr_pair.second);
  }
  if (index > 0) {
    --index;
    for (auto output : outputs_) {
      output->Folder(op_desc, outputs, index, change, begin_node);
    }
  } else {
    for (auto &op_output : this->op_desc_->outputs_) {
      op_desc->outputs_.emplace(op_output.first, op_output.second);
    }

    for (auto &output : this->outputs_) {
      auto iter =
          std::find(output->inputs_.begin(), output->inputs_.end(), this);

      if (iter != output->inputs_.end()) {
        output->inputs_.erase(iter);
      }
      output->inputs_.push_back(begin_node);
      outputs->push_back(output);
    }
  }
}

void Node::Description() {
  if (op_desc_.get()) {
    DLOG << *op_desc_;
  } else {
    DLOG << " null ";
  }
}

Print &operator<<(Print &printer, const Node &node) {
  printer << node.ToString();
  return printer;
}

}  // namespace framework
}  // namespace paddle_mobile
