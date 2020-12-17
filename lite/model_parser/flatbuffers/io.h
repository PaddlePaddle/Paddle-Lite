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
#include <set>
#include <string>
#include <vector>
#include "lite/core/scope.h"
#include "lite/core/variable.h"
#include "lite/model_parser/flatbuffers/param_desc.h"
#include "lite/model_parser/flatbuffers/program_desc.h"

namespace paddle {
namespace lite {
namespace fbs {

void FillParam(const std::string& name,
               const lite::Tensor& tensor,
               ParamDescWriteAPI* prog);

void FillTensor(lite::Tensor* tensor, const ParamDescReadAPI& param);

#ifdef LITE_WITH_FLATBUFFERS_DESC
class ParamSerializer {
 public:
  explicit ParamSerializer(model_parser::ByteWriter* writer,
                           uint16_t version = 0)
      : writer_(writer), version_{version}, buf_(new model_parser::Buffer) {
    CHECK(writer_)
        << "A valid writer should be passed in the ctor of param serializer.";
    WriteHeader();
  }
  void ForwardWrite(const lite::Scope& scope,
                    const std::set<std::string>& param_names);

 private:
  void WriteHeader();
  model_parser::ByteWriter* writer_{nullptr};
  uint16_t version_{0};
  std::unique_ptr<model_parser::Buffer> buf_;
};
#endif

class ParamDeserializer {
 public:
  explicit ParamDeserializer(model_parser::ByteReader* reader)
      : reader_(reader), buf_(new model_parser::Buffer) {
    CHECK(reader_)
        << "A valid reader should be passed in the ctor of param deserializer.";
    ReadHeader();
  }
  void ForwardRead(lite::Scope* scope);

 private:
  void ReadBytesToBuffer(size_t size) {
    buf_->ResetLazy(size);
    reader_->Read(buf_->data(), size);
  }
  void ReadHeader();
  model_parser::ByteReader* reader_{nullptr};
  std::unique_ptr<model_parser::Buffer> buf_;
};

namespace deprecated {
void SetScopeWithCombinedParams(lite::Scope* scope,
                                const CombinedParamsDescReadAPI& params);
void SetCombinedParamsWithScope(const lite::Scope& scope,
                                const std::set<std::string>& param_names,
                                CombinedParamsDescWriteAPI* params);
}  // namespace deprecated

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
