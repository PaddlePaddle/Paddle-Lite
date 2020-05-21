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

#include "lite/kernels/xpu/bridges/graph.h"
#include <utility>
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

int Graph::Add(const std::string& name, std::shared_ptr<Node> node) {
  auto it = nodes_.find(name);
  if (it != nodes_.end()) {
    // Only variable node can be shared with the same name
    if (!node->is_var() || !it->second.back()->is_var()) {
      LOG(FATAL) << "[XPU] Const or data node " << name << " is redefined.";
      return -1;
    }
  } else {
    auto ret = nodes_.insert(
        std::make_pair(name, std::vector<std::shared_ptr<Node>>()));
    CHECK(ret.second);
    it = ret.first;
  }
  it->second.push_back(node);
  return it->second.size();
}

// Variable node
std::shared_ptr<Node> Graph::Add(const std::string& name,
                                 const xtcl::xExpr& layer,
                                 PrecisionType precision,
                                 DataLayoutType layout) {
  auto node = std::make_shared<Node>(precision, layout, Node::Role::kVar);
  auto idx = Add(name, node);
  CHECK_GE(idx, 1);
  node->set_data(std::make_shared<xtcl::xExpr>(layer));
  // Generate a unique name for the current XTCL layer
  builder_.SetLayer(name + "__" + paddle::lite::to_string(idx));
  return node;
}

// Const or data node
std::shared_ptr<Node> Graph::Add(const std::string& name,
                                 const Tensor& tensor,
                                 std::vector<int64_t> shape,
                                 DataLayoutType layout) {
  std::shared_ptr<Node> node = nullptr;
  PrecisionType precision = tensor.precision();
  if (tensor.persistable()) {
    // Const node
    node = std::make_shared<Node>(precision, layout, Node::Role::kConst);
    auto idx = Add(name, node);
    CHECK_EQ(idx, 1);
    node->set_data(std::make_shared<xtcl::xExpr>(builder_.CreateTensor(
        name, CvtShape<xtcl::xIndexExpr>(shape), CvtPrecisionType(precision))));
    params_.emplace(std::make_pair(name, *CvtTensor(tensor, shape, layout)));
  } else {
    // Data node
    node = Add(name, shape, precision, layout);
  }
  return node;
}

// Data node
std::shared_ptr<Node> Graph::Add(const std::string& name,
                                 std::vector<int64_t> shape,
                                 PrecisionType precision,
                                 DataLayoutType layout) {
  auto node = std::make_shared<Node>(precision, layout, Node::Role::kData);
  auto idx = Add(name, node);
  CHECK_EQ(idx, 1);
  node->set_data(std::make_shared<xtcl::xExpr>(builder_.CreateTensor(
      name, CvtShape<xtcl::xIndexExpr>(shape), CvtPrecisionType(precision))));
  return node;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
