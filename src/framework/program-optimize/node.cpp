/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include <sstream>

#include "node.h"

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

std::string Node::ToString(std::string blank, const Node *node) const {
  std::stringstream ss;
  ss << type_ << "-> \n";

  if (inputs_.size() > 1 && node != inputs_.back()) {
    return ss.str();
  } else if (inputs_.size() > 1 && node == inputs_.back()) {
    ss << "\n" << blank << type_ << "\n";
  }

  for (int i = 0; i < outputs_.size(); ++i) {
    ss << blank << outputs_[i]->ToString(blank + " ", this) << "";
  }
  return ss.str();
}

std::string Node::ToString() const { return this->ToString(" ", this); }

Node &Node::To(int index) {
  if (index == 0) {
    this->outputs_.clear();
  }

  for (int j = 0; j < this->outputs_.size(); ++j) {
    outputs_[j]->To(index - 1);
  }
  return *this;
}

uint Node::depth(uint begin) {
  uint depth = 0;
  begin++;
  for (int i = 0; i < outputs_.size(); ++i) {
    uint output_depth = outputs_[i]->depth(begin);
    depth = output_depth > depth ? output_depth : depth;
  }
  return begin > depth ? begin : depth;
}

Print &operator<<(Print &printer, const Node &node) {
  printer << node.ToString();
  return printer;
}

} // namespace framework
} // namespace paddle_mobile
