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

#include "framework/program/program-optimize/node.h"
#include <algorithm>
#include <map>
#include <memory>
#include "framework/operator.h"

namespace paddle_mobile {

namespace framework {

std::vector<Node *> Node::operator[](int index) {
  std::vector<Node *> nodes;
  GetNodesWithLocation(index, 0, &nodes);
  return nodes;
}

void Node::GetNodesWithLocation(int index, int now_index,
                                std::vector<Node *> *nodes) {
  if (index == now_index) {
    nodes->push_back(this);
  }

  for (int i = 0; i < this->outputs_.size(); ++i) {
    this->outputs_[i]->GetNodesWithLocation(index, now_index + 1, nodes);
  }
}

Node &Node::operator>(std::shared_ptr<Node> node) {
  outputs_.push_back(node);
  node->inputs_.push_back(this);
  return *node;
}

bool Node::operator==(const Node &in) {
  if (in.type_ == this->type_) {
    if (this->outputs_.size() == in.outputs_.size()) {
      for (int i = 0; i < outputs_.size(); ++i) {
        if (!(this->outputs_[i]->MedianEqual(*in.outputs_[i]))) {
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

bool Node::MedianEqual(const Node &in) {
  if (in.type_ == this->type_) {
    if (this->outputs_.size() == in.outputs_.size()) {
      //      if (this->inputs_.size() != in.inputs_.size()) {
      //        DLOG << " == - this input size: " << this->inputs_.size();
      //        DLOG << " == - ptr of this " << this;
      //        DLOG << " == - in input size: " << in.inputs_.size();
      //        DLOG << " == - input size not equal ";
      //        return false;
      //      } else {
      //        for (int i = 0; i < this->inputs_.size(); ++i) {
      //          if (this->inputs_[i]->type_ != in.inputs_[i]->type_) {
      //            DLOG << " == - input type not equal ";
      //            return false;
      //          }
      //        }
      //      }

      for (int i = 0; i < outputs_.size(); ++i) {
        if (!((*outputs_[i]).MedianEqual(*in.outputs_[i]))) {
          return false;
        }
      }
    } else {
      //      DLOG << " == - output size not equal ";
      return false;
    }
  } else {
    //    DLOG << " == - median type is not equal ";
    return false;
  }
  return true;
}

std::map<std::string, Node *> Node::Relationship() {
  std::map<std::string, Node *> map;
  RelationshipPrivate(&map);
  return map;
}

void Node::RelationshipPrivate(std::map<std::string, Node *> *map) {
  for (auto output : op_desc_->outputs_) {
    for (auto output_key : output.second) {
      (*map)[output_key] = this;
    }
  }
  for (auto output : this->outputs_) {
    output->RelationshipPrivate(map);
  }
}

std::shared_ptr<Node> Node::To(int size) {
  std::shared_ptr<Node> node = std::make_shared<Node>();
  this->To(size - 1, node);
  return node;
}

void Node::To(int index, std::shared_ptr<Node> node) {
  node->op_desc_ = this->op_desc_;
  node->type_ = this->type_;
  node->inputs_ = this->inputs_;
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

int Node::Depth(int begin) {
  int depth = 0;
  begin++;
  for (int i = 0; i < outputs_.size(); ++i) {
    int output_depth = outputs_[i]->Depth(begin);
    depth = output_depth > depth ? output_depth : depth;
  }
  return begin > depth ? begin : depth;
}

Node &Node::Folder(
    int size, std::string type,
    std::map<std::string, std::vector<std::pair<std::string, std::string>>>
        change,
    std::vector<std::shared_ptr<Node>> *removed_nodes) {
  std::shared_ptr<framework::OpDesc> op_desc =
      std::make_shared<framework::OpDesc>();
  op_desc->inputs_ = this->op_desc_->inputs_;
  std::vector<std::shared_ptr<Node>> outputs;
  this->Folder(op_desc, &outputs, size - 1, &change, this, removed_nodes);
  this->outputs_ = outputs;
  this->type_ = type;
  this->op_desc_ = op_desc;
  this->op_desc_->type_ = type;
  return *this;
}

void Node::Folder(
    std::shared_ptr<framework::OpDesc> op_desc,
    std::vector<std::shared_ptr<Node>> *outputs, int index,
    std::map<std::string, std::vector<std::pair<std::string, std::string>>>
        *change,
    Node *begin_node, std::vector<std::shared_ptr<Node>> *removed_nodes) {
  if (change->find(this->type_) != change->end()) {
    auto change_pairs = (*change)[this->type_];
    for (const auto &change_pair : change_pairs) {
      std::map<std::string, int> f;
      if (this->op_desc_->GetInputs().find(change_pair.first) !=
          this->op_desc_->GetInputs().end()) {
        if (op_desc->GetInputs().find(change_pair.second) !=
            op_desc->GetInputs().end()) {
          for (auto value : this->op_desc_->GetInputs()[change_pair.first]) {
            op_desc->GetInputs()[change_pair.second].push_back(value);
          }
        } else {
          op_desc->GetInputs()[change_pair.second] =
              this->op_desc_->GetInputs()[change_pair.first];
        }
      }
    }
  }

  for (auto &attr_pair : this->op_desc_->attrs_) {
    op_desc->attrs_.emplace(attr_pair.first, attr_pair.second);
  }
  if (index > 0) {
    --index;

    for (auto output : outputs_) {
      if (change->find(this->type_) != change->end()) {
        auto change_pairs = (*change)[this->type_];
        for (const auto &change_pair : change_pairs) {
          std::map<std::string, int> f;
          if (this->op_desc_->GetOutputs().find(change_pair.first) !=
              this->op_desc_->GetOutputs().end()) {
            if (op_desc->GetInputs().find(change_pair.second) !=
                op_desc->GetInputs().end()) {
              for (auto value :
                   this->op_desc_->GetOutputs()[change_pair.first]) {
                op_desc->GetInputs()[change_pair.second].push_back(value);
              }
            } else {
              op_desc->GetInputs()[change_pair.second] =
                  this->op_desc_->GetOutputs()[change_pair.first];
            }
          }
        }
      }

      removed_nodes->push_back(output);
      output->Folder(op_desc, outputs, index, change, begin_node,
                     removed_nodes);
    }
  } else {
    for (auto &op_output : this->op_desc_->outputs_) {
      auto output_key = op_output.first;
      if (change->find(this->type_) != change->end()) {
        const auto change_pairs = (*change)[this->type_];
        for (const auto &target : change_pairs) {
          if (target.first == output_key) {
            output_key = target.second;
          }
        }
      }
      op_desc->outputs_.emplace(output_key, op_output.second);
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
#ifdef PADDLE_MOBILE_DEBUG
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
#endif

}  // namespace framework
}  // namespace paddle_mobile
