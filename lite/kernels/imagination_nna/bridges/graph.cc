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

#include "lite/kernels/imagination_nna/bridges/graph.h"
#include <utility>
#include "lite/kernels/imagination_nna/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace imagination_nna {

// Add 1
int Graph::Add(const std::string& name, std::shared_ptr<Node> node) {
  auto it = nodes_.find(name);
  if (it != nodes_.end()) {
    // Only intermediate node can be shared with the same name
    if (!node->is_data() || !it->second.back()->is_data()) {
      LOG(FATAL) << "[NNA] Const or Input node " << name << " is redefined.";
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

// Add 2
std::shared_ptr<Node> Graph::Add(const std::string& name,
                                 const void* const const_data,
                                 std::vector<int64_t> shape,
                                 const TensorInfo& qnt,
                                 Node::Role role) {
  auto node = std::make_shared<Node>(qnt.type, qnt.layout, role);
  auto idx = Add(name, node);
  CHECK_GE(idx, 1);

  imgdnn_tensor_descriptor desc;
  desc.type = qnt.type;
  desc.dimensions = (unsigned)shape.size();
  for (uint32_t i = 0; i < shape.size(); ++i) desc.size[i] = shape[i];

  switch (qnt.type) {
    case IMGDNN_TYPE_F32:
    case IMGDNN_TYPE_I32:
      break;
    case IMGDNN_TYPE_Q_I8:
    case IMGDNN_TYPE_Q_U8:
      desc.quant_param.scale = qnt.scales[0];
      desc.quant_param.zero_point = qnt.zero_points[0];
      break;
    case IMGDNN_TYPE_QPA_I8:
    case IMGDNN_TYPE_QPA_U8:
      desc.quant_param.per_axis = imgdnnCreatePerAxisQuantParam(
          qnt.axis, qnt.count, qnt.scales.data(), qnt.zero_points.data());
      CHECK(desc.quant_param.per_axis != nullptr);
      break;
    default:
      LOG(FATAL) << "[NNA] invalid tensor type set in node: " << name;
      return nullptr;
  }

  imgdnn_tensor out_tensor;
  if (role == Node::Role::kConst) {
    out_tensor = pImgdnnMgr->CreateFixedInputTensor(&desc, const_data, true);
  } else {
    LOG(INFO) << "[NNA] invald role set in this path: " << name;
  }

  if ((desc.type == IMGDNN_TYPE_QPA_I8 || desc.type == IMGDNN_TYPE_QPA_U8) &&
      desc.quant_param.per_axis != nullptr)
    imgdnnDestroyPerAxisQuantParam(desc.quant_param.per_axis);

  node->set_data(out_tensor);

  return node;
}

// Add 3
std::shared_ptr<Node> Graph::Add(const std::string& name,
                                 const Tensor& tensor,
                                 std::vector<int64_t> shape,
                                 const TensorInfo& qnt,
                                 Node::Role role) {
  auto node = std::make_shared<Node>(qnt.type, qnt.layout, role);
  auto idx = Add(name, node);
  CHECK_GE(idx, 1);

  imgdnn_tensor_descriptor desc;
  desc.type = qnt.type;
  desc.dimensions = (unsigned)shape.size();
  for (uint32_t i = 0; i < shape.size(); ++i) desc.size[i] = shape[i];

  switch (qnt.type) {
    case IMGDNN_TYPE_F32:
    case IMGDNN_TYPE_I32:
      break;
    case IMGDNN_TYPE_Q_I8:
    case IMGDNN_TYPE_Q_U8:
      desc.quant_param.scale = qnt.scales[0];
      desc.quant_param.zero_point = qnt.zero_points[0];
      break;
    case IMGDNN_TYPE_QPA_I8:
    case IMGDNN_TYPE_QPA_U8:
      desc.quant_param.per_axis = imgdnnCreatePerAxisQuantParam(
          qnt.axis, qnt.count, qnt.scales.data(), qnt.zero_points.data());
      CHECK(desc.quant_param.per_axis != nullptr);
      break;
    default:
      LOG(FATAL) << "[NNA] invalid tensor type set in node: " << name;
      return nullptr;
  }

  imgdnn_tensor out_tensor;
  if (role == Node::Role::kInput) {
    out_tensor = pImgdnnMgr->CreateInputTensor(&desc);
  } else if (role == Node::Role::kConst) {
    const void* const_data = tensor.raw_data();
    out_tensor = pImgdnnMgr->CreateFixedInputTensor(&desc, const_data, false);
  } else {
    LOG(INFO) << "[NNA] invald role set in this path: " << name;
  }

  if ((desc.type == IMGDNN_TYPE_QPA_I8 || desc.type == IMGDNN_TYPE_QPA_U8) &&
      desc.quant_param.per_axis != nullptr)
    imgdnnDestroyPerAxisQuantParam(desc.quant_param.per_axis);

  node->set_data(out_tensor);

  return node;
}

}  // namespace imagination_nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
