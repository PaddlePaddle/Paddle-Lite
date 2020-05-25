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

#include "lite/kernels/rknpu/bridges/graph.h"
#include <rknpu/graph.h>
#include "lite/kernels/rknpu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace rknpu {

int Graph::Add(const std::string& name, std::shared_ptr<Node> node) {
  auto it = nodes_.find(name);
  if (it != nodes_.end()) {
    // Only variable node can be shared with the same name
    if (!node->is_var() || !it->second.back()->is_var()) {
      LOG(FATAL) << "[RKNPU] Const or data node " << name << " is redefined.";
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
                                 PrecisionType precision,
                                 DataLayoutType layout,
                                 const QuantizationInfo& qnt) {
  std::shared_ptr<Node> node = nullptr;

  if (precision == PrecisionType::kUnk) {
    precision = tensor.precision();  // todo
  }

  if (precision == PrecisionType::kUnk) {
    if (qnt.enable_int8 && qnt.quant_bits == 8) {
      precision = PrecisionType::kInt8;
    } else if (!qnt.enable_int8) {
      precision = PrecisionType::kFloat;
    } else {
      LOG(ERROR) << "[rknpu]:Graph:: tensor precision unknown!";
    }
  }

  if (precision != tensor.precision()) {
    LOG(INFO) << "[rknpu]:Graph::Add: tensor precision mismatch!" << name << ":"
              << PrecisionToStr(precision) << " vs "
              << PrecisionToStr(tensor.precision());
  }

  if (tensor.persistable()) {
    // Const node
    node = std::make_shared<Node>(precision, layout, Node::Role::kConst);
    auto idx = Add(name, node);
    CHECK_EQ(idx, 1);
    auto attr = std::make_shared<rk::nn::TensorAttr>();
    attr->precision = ToRknpuPrecisionType(precision);
    attr->layout = ToRknpuDataLayoutType(layout);
    attr->role = rk::nn::TensorRole::CONST;
    attr->name = name;

    switch (precision) {
      case PrecisionType::kInt8:
        attr->qntBits = 8;
        attr->qntType = rk::nn::QuantizationType::SYMMETRIC;
        attr->qntParamSymmetric.scale = qnt.scale;
        break;
      case PrecisionType::kInt32:
        attr->qntBits = 32;
        attr->qntType = rk::nn::QuantizationType::SYMMETRIC;
        attr->qntParamSymmetric.scale = qnt.scale;
        break;
      default:
        break;
    }

    attr->dims.resize(shape.size());
    for (int i = 0; i < shape.size(); i++) {
      attr->dims[i] = shape[i];
    }

    LOG(INFO) << "[rknpu]:Graph::Add const node:" << name
              << " precision: " << PrecisionToStr(precision)
              << " layout: " << DataLayoutToStr(layout);
    node->set_data(
        rgraph_->CreateTensor(attr, const_cast<void*>(tensor.raw_data())));
  } else {
    // Data node
    node = Add(name, shape, precision, layout, qnt);
  }
  return node;
}

// Data node
std::shared_ptr<Node> Graph::Add(const std::string& name,
                                 std::vector<int64_t> shape,
                                 PrecisionType precision,
                                 DataLayoutType layout,
                                 const QuantizationInfo& qnt) {
  auto node = std::make_shared<Node>(precision, layout, Node::Role::kData);
  auto idx = Add(name, node);
  CHECK_EQ(idx, 1);
  auto attr = std::make_shared<rk::nn::TensorAttr>();
  attr->precision = ToRknpuPrecisionType(precision);
  attr->layout = ToRknpuDataLayoutType(layout);
  attr->role = rk::nn::TensorRole::VAR;
  attr->name = name;

  switch (precision) {
    case PrecisionType::kInt8:
      attr->qntBits = 8;
      attr->qntType = rk::nn::QuantizationType::SYMMETRIC;
      attr->qntParamSymmetric.scale = qnt.scale;
      break;
    case PrecisionType::kInt32:
      attr->qntBits = 32;
      attr->qntType = rk::nn::QuantizationType::SYMMETRIC;
      attr->qntParamSymmetric.scale = qnt.scale;
      break;

    default:
      break;
  }

  attr->dims.resize(shape.size());
  for (int i = 0; i < shape.size(); i++) {
    attr->dims[i] = shape[i];
  }

  LOG(INFO) << "[rknpu]:Graph::Add data node:" << name
            << " precision: " << PrecisionToStr(precision)
            << " layout: " << DataLayoutToStr(layout);
  node->set_data(rgraph_->CreateTensor(attr, nullptr));  // todo
  return node;
}

Graph::Graph() {
  rgraph_ = new rk::nn::Graph();
  CHECK(rgraph_ != nullptr);
}

Graph::~Graph() {}

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
