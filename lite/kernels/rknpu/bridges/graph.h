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

#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"
#include "rknpu/rknpu_pub.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace rknpu {

// Graph and node is defined to collect all of converted RKNPU IR nodes
struct QuantizationInfo {
  int enable_int8;
  int quant_bits;
  std::vector<float> scale;
};

class Node {
 public:
  enum class Role {
    kVar = 0,
    kConst,
    kData,
  };

  Node(std::shared_ptr<rk::nn::Tensor> data,
       PrecisionType precision,
       DataLayoutType layout,
       Role role)
      : data_(data), precision_(precision), layout_(layout), role_(role) {}
  Node(PrecisionType precision, DataLayoutType layout, Role role)
      : precision_(precision), layout_(layout), role_(role) {}

  void set_data(std::shared_ptr<rk::nn::Tensor> data) { data_ = data; }
  void set_precision(PrecisionType precision) { precision_ = precision; }
  void set_layout(DataLayoutType layout) { layout_ = layout; }
  void set_role(Role role) { role_ = role; }
  void set_quant_param(const QuantizationInfo& qnt) { qnt_ = qnt; }

  std::shared_ptr<rk::nn::Tensor> data() { return data_; }
  PrecisionType precision() const { return precision_; }
  DataLayoutType layout() const { return layout_; }
  Role role() const { return role_; }
  bool is_var() const { return role_ == Role::kVar; }
  bool is_const() const { return role_ == Role::kConst; }
  bool is_data() const { return role_ == Role::kData; }

 private:
  std::shared_ptr<rk::nn::Tensor> data_{nullptr};
  PrecisionType precision_{PRECISION(kFloat)};
  DataLayoutType layout_{DATALAYOUT(kNCHW)};
  Role role_{Role::kVar};
  QuantizationInfo qnt_;
};

class Graph {
 public:
  Graph();
  ~Graph();

 public:
  int Add(const std::string& name, std::shared_ptr<Node> node);

  // Const or data node
  std::shared_ptr<Node> Add(const std::string& name,
                            const Tensor& tensor,
                            std::vector<int64_t> shape,
                            PrecisionType precision = PRECISION(kUnk),
                            DataLayoutType layout = DATALAYOUT(kNCHW),
                            const QuantizationInfo& qnt = QuantizationInfo());
  std::shared_ptr<Node> Get(const std::string& name) {
    CHECK(Has(name)) << "[RKNPU] Node " << name << " not found.";
    return nodes_.at(name).back();
  }

  std::shared_ptr<Node> Add(const std::string& name,
                            const Tensor& tensor,
                            PrecisionType precision = PRECISION(kUnk),
                            DataLayoutType layout = DATALAYOUT(kNCHW),
                            const QuantizationInfo& qnt = QuantizationInfo()) {
    return Add(name, tensor, tensor.dims().Vectorize(), precision, layout, qnt);
  }

  // Data node
  std::shared_ptr<Node> Add(const std::string& name,
                            std::vector<int64_t> shape,
                            PrecisionType precision = PRECISION(kFloat),
                            DataLayoutType layout = DATALAYOUT(kNCHW),
                            const QuantizationInfo& qnt = QuantizationInfo());

  std::shared_ptr<Node> Add(const std::string& name,
                            DDim dims,
                            PrecisionType precision = PRECISION(kFloat),
                            DataLayoutType layout = DATALAYOUT(kNCHW),
                            const QuantizationInfo& qnt = QuantizationInfo()) {
    return Add(name, dims.Vectorize(), precision, layout, qnt);
  }

  bool Has(const std::string& name) {
    return nodes_.find(name) != nodes_.end();
  }

  rk::nn::Graph* GetHandle() { return rgraph_; }

 private:
  std::map<std::string, std::vector<std::shared_ptr<Node>>> nodes_;
  rk::nn::Graph* rgraph_;
};

}  // namespace rknpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
