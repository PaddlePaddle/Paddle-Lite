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

// Const node
std::shared_ptr<ge::op::Const> Graph::AddNode(const std::string& name,
                                              const Tensor& tensor,
                                              std::vector<int64_t> shape,
                                              PrecisionType precision,
                                              DataLayoutType layout) {
  auto node = AddNode<ge::op::Const>(name, precision, layout);
  node->set_attr_value(CvtTensor(tensor, shape, precision, layout));
  return node;
}

// Data node
std::shared_ptr<ge::op::Data> Graph::AddNode(const std::string& name,
                                             std::vector<int64_t> shape,
                                             PrecisionType precision,
                                             DataLayoutType layout) {
  auto node = AddNode<ge::op::Data>(name);
  ge::TensorDesc desc(
      ge::Shape(shape), CvtDataLayoutType(layout), CvtPrecisionType(precision));
  node->update_input_desc_x(desc);
  return node;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
