// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include <vector>
#include "lite/model_parser/base/var_desc.h"
#include "lite/model_parser/flatbuffers/framework_generated.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace fbs {

class VarDesc : public VarDescAPI {
 public:
  explicit VarDesc(proto::VarDesc const* desc) : desc_(desc) {}

  std::string Name() const override { return desc_->name()->str(); }

  VarDescAPI::Type GetType() const override {
    return static_cast<VarDescAPI::Type>(desc_->type()->type());
  }

  bool Persistable() const override { return desc_->persistable(); }

  std::vector<int64_t> GetShape() const override {
    CHECK(GetType() == VarDescAPI::Type::LOD_TENSOR);
    const auto& dims = desc_->type()->lod_tensor()->tensor()->dims();
    std::vector<int64_t> dims_vec;
    dims_vec.reserve(dims->size());
    for (const auto& dim : *dims) {
      dims_vec.push_back(dim);
    }
    return dims_vec;
  }

  VarDescAPI::Type GetDataType() const {
    CHECK(GetType() == VarDescAPI::Type::LOD_TENSOR);
    return static_cast<VarDescAPI::Type>(
        desc_->type()->lod_tensor()->tensor()->data_type());
  }

 private:
  proto::VarDesc const* desc_;

  // To reduce overhead, we expect to use namespace aliasing to make cpp::Desc
  // and flatbuffers::Desc replace each other. However, there is no direct
  // inheritance relationship between the two data types, and the read-only
  // version of flatbuffers lacks some write implementations. Therefore, at
  // present, we are temporarily providing a default interface that triggers
  // execution-time errors to avoid type ambiguity and compile-time errors
  // caused by different building options.

 public:
  VarDesc() { NotImplemented(); }
  void SetDataType(Type data_type) { NotImplemented(); }
  void SetShape(const std::vector<int64_t>& dims) { NotImplemented(); }

 private:
  void NotImplemented() const {
    LOG(FATAL) << "The additional interfaces of VarDesc is temporarily "
                  "unavailable in read-only mode.";
  }
  std::vector<int64_t> shape_;
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
