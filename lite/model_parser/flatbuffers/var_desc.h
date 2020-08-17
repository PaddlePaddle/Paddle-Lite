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
#include "lite/model_parser/flatbuffers/traits.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace fbs {

class VarDescView : public VarDescAPI {
 public:
  explicit VarDescView(proto::VarDesc const* desc) : desc_(desc) {}

  std::string Name() const override { return desc_->name()->str(); }

  VarDescAPI::Type GetType() const override {
    return ConvertVarType(desc_->type()->type());
  }

  bool Persistable() const override { return desc_->persistable(); }

  std::vector<int64_t> GetShape() const override {
    CHECK(GetType() == VarDescAPI::Type::LOD_TENSOR);
    const auto& dims = desc_->type()->lod_tensor()->tensor()->dims();
    std::vector<int64_t> dims_vec;
    dims_vec.resize(dims->size());
    for (size_t i = 0; i < dims->size(); ++i) {
      dims_vec[i] = dims->operator[](i);
    }
    return dims_vec;
  }

  VarDescAPI::Type GetDataType() const {
    CHECK(GetType() == VarDescAPI::Type::LOD_TENSOR);
    return ConvertVarType(desc_->type()->lod_tensor()->tensor()->data_type());
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
  VarDescView() = default;
  void SetDataType(Type data_type) { NotImplemented(); }
  void SetShape(const std::vector<int64_t>& dims) { NotImplemented(); }

 private:
  void NotImplemented() const {
    LOG(FATAL) << "The additional interfaces of VarDescView is temporarily "
                  "unavailable in read-only mode.";
  }
  std::vector<int64_t> shape_;
};

class VarDesc : public VarDescAPI {
 public:
  VarDesc() : owned_(true), desc_(new proto::VarDescT()) {}

  explicit VarDesc(proto::VarDescT* desc) : desc_(desc) {
    CHECK(desc_);
    InitType();
  }

  std::string Name() const override { return desc_->name; }

  void SetName(std::string name) override { desc_->name = name; }

  Type GetType() const override { return ConvertVarType(type_->type); }

  void SetType(Type type) override { type_->type = ConvertVarType(type); }

  void SetDataType(Type type) {
    type_->lod_tensor->tensor->data_type = ConvertVarType(type);
  }

  Type GetDataType() const {
    return ConvertVarType(type_->lod_tensor->tensor->data_type);
  }

  bool Persistable() const override { return desc_->persistable; }

  void SetPersistable(bool persistable) override {
    desc_->persistable = persistable;
  }

  std::vector<int64_t> GetShape() const override {
    CHECK(GetType() == VarDescAPI::Type::LOD_TENSOR);
    return type_->lod_tensor->tensor->dims;
  }

  void SetShape(const std::vector<int64_t>& dims) override {
    type_->lod_tensor->tensor->dims = dims;
  }

  proto::VarDescT* raw_desc() { return desc_; }

  ~VarDesc() {
    if (owned_) {
      delete desc_;
    }
  }

 private:
  void InitType() {
    if (!desc_->type) {
      desc_->type = std::unique_ptr<proto::VarTypeT>(new proto::VarTypeT());
      desc_->type->lod_tensor =
          std::unique_ptr<proto::VarType_::LoDTensorDescT>(
              new proto::VarType_::LoDTensorDescT());
      desc_->type->lod_tensor->tensor =
          std::unique_ptr<proto::VarType_::TensorDescT>(
              new proto::VarType_::TensorDescT());
    }
    type_ = desc_->type.get();
  }
  bool owned_{false};
  proto::VarDescT* desc_{nullptr};
  paddle::lite::fbs::proto::VarTypeT* type_{nullptr};
};

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
