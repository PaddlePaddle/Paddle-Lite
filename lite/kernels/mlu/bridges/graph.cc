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

#include "lite/kernels/mlu/bridges/graph.h"
#include <utility>
#include <vector>
#include "lite/kernels/mlu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

std::shared_ptr<MLUTensor> Graph::AddNode(const std::string& name,
                                          std::vector<int64_t> shape,
                                          cnmlTensorType_t tensor_type,
                                          cnmlDataOrder_t shape_order,
                                          cnmlDataType_t mlu_dtype,
                                          cnmlDataOrder_t data_order,
                                          void* raw_ptr) {
  CHECK(!HasNode(name));
  VLOG(5) << "add mlu node: " << name << "\t data type "
          << static_cast<int>(mlu_dtype) << "\t data order "
          << static_cast<int>(data_order);
  auto node = std::shared_ptr<MLUTensor>(
      new MLUTensor(shape, tensor_type, shape_order, mlu_dtype, data_order));
  node->set_mlu_ptr(raw_ptr);
  nodes_.insert(std::make_pair(name, node));
  return node;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
