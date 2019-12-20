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

#include <xtcl/xtcl.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace xpu {

// The Context of the converters which used for converting the ops of subgraph
// to the XPU IR graph
class Graph {
 public:
  // Layer node
  std::shared_ptr<xtcl::xExpr> AddNode(const std::string& name,
                                       const xtcl::xExpr& layer);

  // Const node
  std::shared_ptr<xtcl::xExpr> AddNode(
      const std::string& name,
      const Tensor& tensor,
      PrecisionType ptype = PRECISION(kFloat),
      DataLayoutType ltype = DATALAYOUT(kNCHW));

  std::shared_ptr<xtcl::xExpr> AddNode(
      const std::string& name,
      const Tensor& tensor,
      std::vector<int64_t> shape,
      PrecisionType ptype = PRECISION(kFloat),
      DataLayoutType ltype = DATALAYOUT(kNCHW));

  template <typename T>
  std::shared_ptr<xtcl::xExpr> AddNode(
      const std::string& name,
      const std::vector<T>& data,
      std::vector<int64_t> shape = {},
      DataLayoutType ltype = DATALAYOUT(kNCHW)) {
    const std::type_info& info = typeid(T);
    PrecisionType ptype = PRECISION(kFloat);
    if (info == typeid(float)) {
      ptype = PRECISION(kFloat);
    } else if (info == typeid(int8_t)) {
      ptype = PRECISION(kFloat);
    } else if (info == typeid(int32_t)) {
      ptype = PRECISION(kInt32);
    } else {
      LOG(FATAL) << "[XPU] Unknow data type " << info.name();
    }
    if (shape.empty()) {
      shape = {static_cast<int64_t>(data.size())};
    } else {
      int size = 1;
      for (auto i : shape) {
        size *= i;
      }
      CHECK_EQ(data.size(), size);
    }
    Tensor tensor;
    tensor.Resize(shape);
    std::memcpy(reinterpret_cast<uint8_t*>(tensor.mutable_data<T>()),
                reinterpret_cast<const uint8_t*>(data.data()),
                data.size() * sizeof(T));
    return AddNode(name, tensor, ptype, ltype);
  }

  template <typename T>
  std::shared_ptr<xtcl::xExpr> AddNode(
      const std::string& name,
      T value,
      std::vector<int64_t> shape = {1},
      DataLayoutType ltype = DATALAYOUT(kNCHW)) {
    int64_t size = 1;
    for (auto i : shape) {
      size *= i;
    }
    std::vector<T> data(size, value);
    return AddNode(name, data, shape, ltype);
  }

  // Data node
  std::shared_ptr<xtcl::xExpr> AddNode(
      const std::string& name,
      std::vector<int64_t> shape,
      PrecisionType ptype = PRECISION(kFloat),
      DataLayoutType ltype = DATALAYOUT(kNCHW));

  std::shared_ptr<xtcl::xExpr> GetNode(const std::string& name) {
    CHECK(HasNode(name)) << "[XPU] Node " << name << " not found.";
    return nodes_.at(name);
  }

  bool HasNode(const std::string& name) {
    return nodes_.find(name) != nodes_.end();
  }

 public:
  // XPU network builder and constant tensors
  xtcl::network::xNetworkBuilder builder_;
  xtcl::network::xTensorCompiler::ParamNDArrayMap params_;

 private:
  std::unordered_map<std::string, std::shared_ptr<xtcl::xExpr>> nodes_;
  std::unordered_map<std::string, int> counts_;
};

}  // namespace xpu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
