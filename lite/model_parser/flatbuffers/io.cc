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
#include <memory>
#include <utility>
#include <vector>
#include "lite/model_parser/flatbuffers/traits.h"

namespace paddle {
namespace lite {
namespace fbs {

std::vector<char> LoadFile(const std::string& path) {
  FILE* file = fopen(path.c_str(), "rb");
  CHECK(file);
  fseek(file, 0, SEEK_END);
  uint64_t length = ftell(file);
  rewind(file);
  std::vector<char> buf(length);
  CHECK_EQ(fread(buf.data(), 1, length, file), length);
  fclose(file);
  return buf;
}

void SaveFile(const std::string& path, const std::vector<char>& cache) {
  FILE* file = fopen(path.c_str(), "wb");
  CHECK(file);
  CHECK(fwrite(cache.data(), sizeof(char), cache.size(), file) == cache.size());
  fclose(file);
}

void SetParamWithTensor(const std::string& name,
                        const lite::Tensor& tensor,
                        ParamDescWriteAPI* prog) {
  CHECK(prog);
  prog->SetName(name);
  prog->SetDim(tensor.dims().Vectorize());
  prog->SetDataType(lite::ConvertPrecisionType(tensor.precision()));
  prog->SetData(tensor.raw_data(), tensor.memory_size());
}

void SetTensorWithParam(lite::Tensor* tensor, const ParamDescReadAPI& param) {
  tensor->Resize(param.Dim());
  tensor->set_precision(lite::ConvertPrecisionType(param.GetDataType()));
  std::memcpy(tensor->mutable_data(param.byte_size()),
              param.GetData(),
              param.byte_size());
}

void SetCombinedParamsWithScope(const lite::Scope& scope,
                                const std::set<std::string>& params_name,
                                CombinedParamsDescWriteAPI* params) {
  for (const auto& name : params_name) {
    auto* param = params->AddParamDesc();
    auto& tensor = scope.FindVar(name)->Get<lite::Tensor>();
    SetParamWithTensor(name, tensor, param);
  }
}

void SetScopeWithCombinedParams(lite::Scope* scope,
                                const CombinedParamsDescReadAPI& params) {
  CHECK(scope);
  for (size_t i = 0; i < params.GetParamsSize(); ++i) {
    const auto& param = *params.GetParamDesc(i);
    auto* tensor = scope->Var(param.Name())->GetMutable<lite::Tensor>();
    SetTensorWithParam(tensor, param);
  }
}

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
