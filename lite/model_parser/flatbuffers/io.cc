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

#include "lite/model_parser/flatbuffers/io.h"
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/model/base/io.h"
#include "lite/model_parser/flatbuffers/traits.h"

namespace paddle {
namespace lite {
namespace fbs {
namespace deprecated {
void SetCombinedParamsWithScope(const lite::Scope& scope,
                                const std::set<std::string>& param_names,
                                CombinedParamsDescWriteAPI* params) {
  for (const auto& name : param_names) {
    auto* param = params->AddParamDesc();
    auto& tensor = scope.FindVar(name)->Get<lite::Tensor>();
    FillParam(name, tensor, param);
  }
}

void SetScopeWithCombinedParams(lite::Scope* scope,
                                const CombinedParamsDescReadAPI& params) {
  CHECK(scope);
  for (size_t i = 0; i < params.GetParamsSize(); ++i) {
    const auto* param = params.GetParamDesc(i);
    CHECK(param);
    auto* tensor = scope->Var(param->Name())->GetMutable<lite::Tensor>();
    CHECK(tensor);
    FillTensor(tensor, *param);
  }
}
}  // namespace deprecated

void FillParam(const std::string& name,
               const lite::Tensor& tensor,
               ParamDescWriteAPI* prog) {
  CHECK(prog);
  prog->SetName(name);
  prog->SetDim(tensor.dims().Vectorize());
  prog->SetDataType(lite::ConvertPrecisionType(tensor.precision()));
  prog->SetData(tensor.raw_data(), tensor.memory_size());
}

void FillTensor(lite::Tensor* tensor, const ParamDescReadAPI& param) {
  CHECK(tensor);
  tensor->Resize(param.Dim());
  tensor->set_precision(lite::ConvertPrecisionType(param.GetDataType()));
  auto* dst = tensor->mutable_data(param.byte_size());
  CHECK(dst);
  CHECK(param.GetData());
  std::memcpy(dst, param.GetData(), param.byte_size());
  tensor->set_persistable(true);
}
#ifdef LITE_WITH_FLATBUFFERS_DESC
void ParamSerializer::ForwardWrite(const lite::Scope& scope,
                                   const std::set<std::string>& param_names) {
  const uint16_t params_size = param_names.size();
  // meta_information
  uint32_t max_tensor_size = 0;
  for (const auto& name : param_names) {
    auto& tensor = scope.FindVar(name)->Get<lite::Tensor>();
    size_t tensor_size =
        tensor.numel() * lite_api::PrecisionTypeLength(tensor.precision());
    if (tensor_size > max_tensor_size) {
      max_tensor_size = tensor_size;
    }
  }
  constexpr uint16_t header_size =
      sizeof(params_size) + sizeof(max_tensor_size);
  CHECK_LT(max_tensor_size, (std::numeric_limits<uint32_t>::max)())
      << "The size of param is out of range.";

  writer_->Write<uint16_t>(header_size);
  writer_->Write<uint16_t>(params_size);
  writer_->Write<uint32_t>(max_tensor_size);

  for (const auto& name : param_names) {
    fbs::ParamDesc param;
    auto& tensor = scope.FindVar(name)->Get<lite::Tensor>();
    FillParam(name, tensor, &param);
    param.CopyDataToBuffer(buf_.get());

    const size_t param_bytes = buf_->size();
    CHECK(param_bytes) << "The bytes size of param can not be zero";
    constexpr uint32_t offset = sizeof(uint32_t);
    const uint32_t total_size = param_bytes + offset;
    writer_->Write<uint32_t>(total_size);
    writer_->Write<uint32_t>(offset);
    writer_->Write(buf_->data(), param_bytes);
  }
}

void ParamSerializer::WriteHeader() {
  // 1. version id
  writer_->Write<uint16_t>(version_);
  // 2. size of meta information (reserved)
  writer_->Write<uint16_t>(0U);
}
#endif

void ParamDeserializer::ForwardRead(lite::Scope* scope) {
  CHECK(scope) << "The pointer of scope is nullptr";
  uint16_t header_size = reader_->Read<uint16_t>();
  ReadBytesToBuffer(header_size);
  char const* data = static_cast<char const*>(buf_->data());
  uint16_t params_size = *reinterpret_cast<uint16_t const*>(data);
  uint32_t max_tensor_size =
      *reinterpret_cast<uint32_t const*>(data + sizeof(uint16_t));

  buf_->ResetLazy(max_tensor_size);
  for (size_t i = 0; i < params_size; ++i) {
    uint32_t total_size = reader_->Read<uint32_t>();
    uint32_t offset = reader_->Read<uint32_t>();
    uint32_t param_bytes = total_size - offset;
    ReadBytesToBuffer(offset - sizeof(offset));
    ReadBytesToBuffer(param_bytes);
    fbs::ParamDescView param(buf_.get());
    FillTensor(scope->Var(param.Name())->GetMutable<lite::Tensor>(), param);
  }
}

void ParamDeserializer::ReadHeader() {
  // 1. version id
  uint16_t version = reader_->Read<uint16_t>();
  CHECK_EQ(version, 0U)
      << "File format error: The version of params must be zero.";
  // 2. meta version
  uint16_t meta_size = reader_->Read<uint16_t>();
  ReadBytesToBuffer(meta_size);
}

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
