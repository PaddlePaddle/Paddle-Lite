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

#include "lite/kernels/xpu/bridges/context.h"
#include <utility>
#include "lite/kernels/xpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

std::shared_ptr<xtcl::xExpr> Context::AddNode(const std::string& name,
                                              const xtcl::xExpr& layer) {
  auto unique_name = [&](const std::string& key) {
    int idx = 1;
    auto it = counts_.find(key);
    if (it == counts_.end()) {
      counts_.insert(std::make_pair(key, idx));
    } else {
      idx = ++(it->second);
    }
    return key + "_" + std::to_string(idx);
  };
  auto it = nodes_.find(name);
  if (it != nodes_.end()) {
    CHECK(params_.find(name) == params_.end()) << "[XPU] Node " << name
                                               << " redefined.";
    // Generate a new unique name as the key to bind the origin node if the
    // origin node isn't a const node: new_name->node
    nodes_.insert(std::make_pair(unique_name(name + "_var"), it->second));
    nodes_.erase(it);
  }
  // Create a new node and bind with the name: name->new_node
  auto node = std::make_shared<xtcl::xExpr>(layer);
  nodes_.insert(std::make_pair(name, node));
  builder_.SetLayer(unique_name(name + "_op"));
  return node;
}

// Const node
std::shared_ptr<xtcl::xExpr> Context::AddNode(const std::string& name,
                                              const Tensor& tensor,
                                              PrecisionType ptype,
                                              DataLayoutType ltype) {
  return AddNode(name, tensor, tensor.dims().Vectorize(), ptype, ltype);
}

std::shared_ptr<xtcl::xExpr> Context::AddNode(const std::string& name,
                                              const Tensor& tensor,
                                              std::vector<int64_t> shape,
                                              PrecisionType ptype,
                                              DataLayoutType ltype) {
  auto node = AddNode(name, shape, ptype, ltype);
  params_.emplace(
      std::make_pair(name, *CvtTensor(tensor, shape, ptype, ltype)));
  return node;
}

// Data node
std::shared_ptr<xtcl::xExpr> Context::AddNode(const std::string& name,
                                              std::vector<int64_t> shape,
                                              PrecisionType ptype,
                                              DataLayoutType ltype) {
  CHECK(!HasNode(name));
  auto node = std::make_shared<xtcl::xExpr>(
      builder_.CreateTensor(name, CvtShape(shape), CvtPrecisionType(ptype)));
  nodes_.insert(std::make_pair(name, node));
  return node;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
