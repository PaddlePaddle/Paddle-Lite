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

#include "lite/kernels/npu/bridges/graph.h"
#include <utility>
#include "lite/kernels/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int Graph::Add(const std::string& name, std::shared_ptr<Node> node) {
  auto it = nodes_.find(name);
  if (it != nodes_.end()) {
    // Only variable node can be shared with the same name
    if (!node->is_var() || !it->second.back()->is_var()) {
      LOG(FATAL) << "[NPU] Const or data node " << name << " is redefined.";
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

// Const or data node
std::shared_ptr<Node> Graph::Add(const std::string& name,
                                 const Tensor& tensor,
                                 std::vector<int64_t> shape,
                                 DataLayoutType layout) {
  std::shared_ptr<Node> node = nullptr;
  PrecisionType precision = tensor.precision();
  if (tensor.persistable()) {
    // Const node
    node = Add<ge::op::Const>(name, precision, layout);
    node->data<ge::op::Const>()->set_attr_value(
        CvtTensor(tensor, shape, layout));
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
  auto node = Add<ge::op::Data>(name, precision, layout);
  ge::TensorDesc desc(
      ge::Shape(shape), CvtDataLayoutType(layout), CvtPrecisionType(precision));
  node->data<ge::op::Data>()->update_input_desc_x(desc);
  return node;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
