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

std::shared_ptr<xtcl::xExpr> Graph::AddNode(const std::string& name,
                                            const xtcl::xExpr& layer,
                                            PrecisionType precision,
                                            DataLayoutType layout) {
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
    // Only variable can rebind the name
    CHECK(!it->second.second.persistable()) << "[XPU] Node " << name
                                            << " redefined.";
    // Generate a new unique name as the key to bind the origin node if the
    // origin node isn't a const node: new_name->node
    nodes_.insert(std::make_pair(unique_name(name + "_var"), it->second));
    nodes_.erase(it);
  }
  // Create a new node and bind with the name: name->new_node
  auto node = std::make_shared<xtcl::xExpr>(layer);
  nodes_.insert(std::make_pair(
      name, std::make_pair(node, Type(precision, layout, false))));
  builder_.SetLayer(unique_name(name + "_op"));
  return node;
}

// Const node
std::shared_ptr<xtcl::xExpr> Graph::AddNode(const std::string& name,
                                            const Tensor& tensor,
                                            PrecisionType precision,
                                            DataLayoutType layout) {
  return AddNode(name, tensor, tensor.dims().Vectorize(), precision, layout);
}

std::shared_ptr<xtcl::xExpr> Graph::AddNode(const std::string& name,
                                            const Tensor& tensor,
                                            std::vector<int64_t> shape,
                                            PrecisionType precision,
                                            DataLayoutType layout) {
  CHECK(!HasNode(name)) << "[NPU] Node " << name << " redefined.";
  auto node = std::make_shared<xtcl::xExpr>(builder_.CreateTensor(
      name, CvtShape<xtcl::xIndexExpr>(shape), CvtPrecisionType(precision)));
  nodes_.insert(std::make_pair(
      name, std::make_pair(node, Type(precision, layout, true))));
  params_.emplace(
      std::make_pair(name, *CvtTensor(tensor, shape, precision, layout)));
  return node;
}

// Data node
std::shared_ptr<xtcl::xExpr> Graph::AddNode(const std::string& name,
                                            std::vector<int64_t> shape,
                                            PrecisionType precision,
                                            DataLayoutType layout) {
  CHECK(!HasNode(name)) << "[NPU] Node " << name << " redefined.";
  auto node = std::make_shared<xtcl::xExpr>(builder_.CreateTensor(
      name, CvtShape<xtcl::xIndexExpr>(shape), CvtPrecisionType(precision)));
  nodes_.insert(std::make_pair(
      name, std::make_pair(node, Type(precision, layout, false))));
  return node;
}

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
