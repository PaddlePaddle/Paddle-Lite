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

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/model_parser/base/io.h"
#include "lite/model_parser/base/param_desc.h"
#include "lite/model_parser/flatbuffers/framework_generated.h"
#include "lite/model_parser/flatbuffers/param_generated.h"
#include "lite/model_parser/flatbuffers/traits.h"

namespace paddle {
namespace lite {
namespace fbs {

class ParamDescView : public ParamDescReadAPI {
 public:
  explicit ParamDescView(model_parser::Buffer* buf) {
    CHECK(buf) << "The pointer in buf can not be nullptr";
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf->data()),
                                   buf->size());
    CHECK(verifier.VerifyBuffer<paddle::lite::fbs::proto::ParamDesc>(nullptr))
        << "Param verification failed.";
    desc_ =
        flatbuffers::GetRoot<paddle::lite::fbs::proto::ParamDesc>(buf->data());
    Init();
  }
  explicit ParamDescView(proto::ParamDesc const* desc) : desc_(desc) { Init(); }
  void Init() {
    CHECK(desc_);
    CHECK(desc_->variable_type() ==
          proto::ParamDesc_::VariableDesc_LoDTensorDesc);
    tensor_desc_ = desc_->variable_as<proto::ParamDesc_::LoDTensorDesc>();
    CHECK(tensor_desc_);
    CHECK(tensor_desc_->data());
  }
  std::string Name() const override {
    CHECK(desc_->name());
    return desc_->name()->c_str();
  }

  std::vector<int64_t> Dim() const override {
    const auto* dims = tensor_desc_->dim();
    CHECK(dims);
    std::vector<int64_t> dims_vec;
    dims_vec.resize(dims->size());
    for (size_t i = 0; i < dims->size(); ++i) {
      dims_vec[i] = dims->operator[](i);
    }
    return dims_vec;
  }

  VarDataType GetDataType() const override {
    return ConvertVarType(tensor_desc_->data_type());
  }

  const void* GetData() const override { return tensor_desc_->data()->Data(); }

  size_t byte_size() const override { return tensor_desc_->data()->size(); }

  ParamDescView() = default;

 private:
  proto::ParamDesc const* desc_;
  proto::ParamDesc_::LoDTensorDesc const* tensor_desc_;
};

class CombinedParamsDescView : public CombinedParamsDescReadAPI {
 public:
  CombinedParamsDescView() = default;
  explicit CombinedParamsDescView(model_parser::Buffer&& buf) {
    Init(std::forward<model_parser::Buffer>(buf));
  }

  void Init(model_parser::Buffer&& buf) {
    CHECK(buf.data());
    buf_ = std::move(buf);
    InitParams();
  }

  void InitParams() {
    flatbuffers::Verifier verifier(static_cast<const uint8_t*>(buf_.data()),
                                   buf_.size());
    CHECK(verifier.VerifyBuffer<paddle::lite::fbs::proto::CombinedParamsDesc>(
        nullptr))
        << "CombinedParamsDesc verification failed.";
    desc_ = proto::GetCombinedParamsDesc(buf_.data());
    CHECK(desc_);
    CHECK(desc_->params());
    size_t params_size = desc_->params()->size();
    params_.resize(params_size);
    for (size_t idx = 0; idx < params_size; ++idx) {
      params_[idx] = ParamDescView(desc_->params()->Get(idx));
    }
  }

  const ParamDescReadAPI* GetParamDesc(size_t idx) const override {
    CHECK(idx < GetParamsSize());
    return &params_[idx];
  }

  size_t GetParamsSize() const override { return params_.size(); }

 private:
  std::vector<ParamDescView> params_;
  model_parser::Buffer buf_;
  proto::CombinedParamsDesc const* desc_;
};

#ifdef LITE_WITH_FLATBUFFERS_DESC
class ParamDesc : public ParamDescAPI {
 public:
  ParamDesc() : owned_(true), desc_(new proto::ParamDescT()) {
    desc_->variable.Set(proto::ParamDesc_::LoDTensorDescT());
    lod_tensor_ = desc_->variable.AsLoDTensorDesc();
    CHECK(lod_tensor_);
  }

  explicit ParamDesc(proto::ParamDescT* desc) : desc_(desc) {
    if (desc_->variable.type == proto::ParamDesc_::VariableDesc_NONE) {
      desc_->variable.Set(proto::ParamDesc_::LoDTensorDescT());
    }
    CHECK(desc_->variable.type ==
          proto::ParamDesc_::VariableDesc_LoDTensorDesc);
    lod_tensor_ = desc_->variable.AsLoDTensorDesc();
    CHECK(lod_tensor_);
  }

  std::string Name() const override { return desc_->name; }
  void SetName(const std::string& name) override { desc_->name = name; }

  std::vector<int64_t> Dim() const override { return lod_tensor_->dim; }
  void SetDim(const std::vector<int64_t>& dim) override {
    lod_tensor_->dim = dim;
  }

  VarDataType GetDataType() const override {
    return ConvertVarType(lod_tensor_->data_type);
  }
  void SetDataType(VarDataType data_type) override {
    lod_tensor_->data_type = ConvertVarType(data_type);
  }

  const void* GetData() const override { return lod_tensor_->data.data(); }

  size_t byte_size() const override { return lod_tensor_->data.size(); }

  void SetData(const void* data, size_t byte_size) override {
    lod_tensor_->data.resize(byte_size);
    model_parser::memcpy(lod_tensor_->data.data(), data, byte_size);
  }

  const proto::ParamDescT* raw_desc() const { return desc_; }

  void CopyDataToBuffer(model_parser::Buffer* buffer) {
    CHECK(buffer);
    SyncBuffer();
    buffer->ResetLazy(buf_.size());
    model_parser::memcpy(buffer->data(), buf_.data(), buf_.size());
  }

  void SyncBuffer() {
    fbb_.Reset();
    flatbuffers::Offset<proto::ParamDesc> desc =
        proto::ParamDesc::Pack(fbb_, desc_);
    fbb_.Finish(desc);
    buf_ = fbb_.Release();
  }

  size_t GetBufferMinAlignment() { return fbb_.GetBufferMinAlignment(); }

  ~ParamDesc() {
    if (owned_) {
      delete desc_;
    }
  }

 private:
  bool owned_{false};
  proto::ParamDescT* desc_{nullptr};
  proto::ParamDesc_::LoDTensorDescT* lod_tensor_{nullptr};
  flatbuffers::DetachedBuffer buf_;
  flatbuffers::FlatBufferBuilder fbb_;
};

class CombinedParamsDesc : public CombinedParamsDescAPI {
 public:
  CombinedParamsDesc() = default;

  explicit CombinedParamsDesc(const Buffer& buf) {
    const auto* raw_buf = proto::GetCombinedParamsDesc(buf.data());
    raw_buf->UnPackTo(&desc_);
    SyncParams();
  }

  const ParamDescReadAPI* GetParamDesc(size_t idx) const override {
    return params_[idx].get();
  }

  size_t GetParamsSize() const override { return desc_.params.size(); }

  ParamDescWriteAPI* AddParamDesc() override {
    desc_.params.push_back(
        std::unique_ptr<proto::ParamDescT>(new proto::ParamDescT));
    SyncParams();
    return params_[params_.size() - 1].get();
  }

  void CopyDataToBuffer(model_parser::Buffer* buffer) {
    CHECK(buffer);
    SyncBuffer();
    buffer->ResetLazy(buf_.size());
    model_parser::memcpy(buffer->data(), buf_.data(), buf_.size());
  }

  size_t GetBufferMinAlignment() { return fbb_.GetBufferMinAlignment(); }

 private:
  void SyncParams() {
    params_.resize(GetParamsSize());
    for (size_t i = 0; i < GetParamsSize(); ++i) {
      if (!params_[i] || params_[i]->raw_desc() != desc_.params[i].get()) {
        params_[i].reset(new ParamDesc(desc_.params[i].get()));
      }
    }
  }

  void SyncBuffer() {
    fbb_.Reset();
    flatbuffers::Offset<proto::CombinedParamsDesc> desc =
        proto::CombinedParamsDesc::Pack(fbb_, &desc_);
    fbb_.Finish(desc);
    buf_ = fbb_.Release();
  }

  flatbuffers::DetachedBuffer buf_;
  flatbuffers::FlatBufferBuilder fbb_;
  proto::CombinedParamsDescT desc_;
  std::vector<std::unique_ptr<ParamDesc>> params_;
};
#endif  // LITE_WITH_FLATBUFFERS_DESC

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
