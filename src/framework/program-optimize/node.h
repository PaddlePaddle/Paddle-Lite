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

#pragma once

#include <string>
#include <vector>

#include "common/log.h"
#include "framework/op_desc.h"
#include "framework/paddle_mobile_object.h"

namespace paddle_mobile {
namespace framework {

class Node : PaddleMobileObject {
  public:
    Node(const std::string &type) : type_(type) {}
    Node(std::shared_ptr<OpDesc> op_desc)
        : op_desc_(op_desc), type_(op_desc->Type()){};
    Node &operator>(std::shared_ptr<Node> node);
    bool operator==(const Node &in);
    std::string ToString() const;
    Node &To(int index);
    uint depth(uint begin = 0);

  private:
    std::shared_ptr<OpDesc> op_desc_;
    std::string ToString(std::string blank, const Node *node) const;
    std::vector<std::shared_ptr<Node>> outputs_;
    std::vector<Node *> inputs_;
    std::string type_;
};

Print &operator<<(Print &printer, const Node &node);
} // namespace framework
} // namespace paddle_mobile
