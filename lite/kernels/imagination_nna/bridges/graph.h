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

#include <imgdnn.h>
#include <math.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "lite/backends/imagination_nna/imgdnn_manager.h"
#include "lite/core/op_lite.h"
#include "lite/core/tensor.h"
#include "utility.h"  // NOLINT

namespace paddle {
namespace lite {
namespace subgraph {
namespace imagination_nna {

#define NNA_UNUSED(var) \
  do {                  \
    (void)(var);        \
  } while (0)

// Graph and node is defined to collect all of converted IMGDNN IR nodes
class Node {
 public:
  enum class Role {
    kInput = 0,
    kConst,
    kData,
  };

  Node(imgdnn_tensor data, imgdnn_type type, DataLayoutType layout, Role role)
      : data_(data), type_(type), layout_(layout), role_(role) {}

  Node(imgdnn_type type, DataLayoutType layout, Role role)
      : type_(type), layout_(layout), role_(role) {}

  void set_data(imgdnn_tensor data) { data_ = data; }
  void set_type(imgdnn_type type) { type_ = type; }
  void set_layout(DataLayoutType layout) { layout_ = layout; }
  void set_role(Role role) { role_ = role; }

  template <typename T>
  std::shared_ptr<T> data() {
    return std::static_pointer_cast<T>(data_);
  }
  imgdnn_tensor data() { return data_; }
  imgdnn_type type() const { return type_; }
  DataLayoutType layout() const { return layout_; }

  bool is_input() const { return role_ == Role::kInput; }
  bool is_const() const { return role_ == Role::kConst; }
  bool is_data() const { return role_ == Role::kData; }

 private:
  imgdnn_tensor data_{nullptr};
  imgdnn_type type_{IMGDNN_TYPE_MAX};
  DataLayoutType layout_{DATALAYOUT(kNCHW)};
  Role role_{Role::kData};
};

class Graph {
 public:
  explicit Graph(lite::imagination_nna::ImgdnnManager* pMgr) {
    pImgdnnMgr = pMgr;
  }

  // Add constant tensor, such as weights,bias
  std::shared_ptr<Node> Add(const std::string& name,
                            const void* const const_data,
                            std::vector<int64_t> shape,
                            const TensorInfo& qnt,
                            Node::Role role);

  // Add input tensor
  std::shared_ptr<Node> Add(const std::string& name,
                            const Tensor& tensor,
                            const TensorInfo& qnt,
                            Node::Role role) {
    return Add(name, tensor, tensor.dims().Vectorize(), qnt, role);
  }

  // Add intermediate activation tensor
  int Add(const std::string& name,
          imgdnn_tensor img_tensor,
          imgdnn_type type,
          DataLayoutType layout = DATALAYOUT(kNCHW)) {
    Node::Role role = Node::Role::kData;
    auto node = std::make_shared<Node>(type, layout, role);
    node->set_data(img_tensor);
    return Add(name, node);  // call Add 1
  }

  std::shared_ptr<Node> Get(std::string name) {
    CHECK(Has(name)) << "[NNA] Node " << name << " not found.";
    return nodes_.at(name).back();
  }

  bool Has(const std::string& name) {
    return nodes_.find(name) != nodes_.end();
  }

  lite::imagination_nna::ImgdnnManager* GetBuilder() {
    CHECK(pImgdnnMgr != nullptr) << "pImgdnnMgr used before initialize";
    return pImgdnnMgr;
  }

 private:
  int Add(const std::string& name, std::shared_ptr<Node> node);

  std::shared_ptr<Node> Add(const std::string& name,
                            const Tensor& tensor,
                            std::vector<int64_t> shape,
                            const TensorInfo& qnt,
                            Node::Role role);

  std::unordered_map<std::string, std::vector<std::shared_ptr<Node>>> nodes_;
  lite::imagination_nna::ImgdnnManager* pImgdnnMgr{nullptr};
};

}  // namespace imagination_nna
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
