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

#include <algorithm>
#include <string>
#include <vector>
#include "lite/model_parser/desc_apis.h"
#include "lite/model_parser/naive_buffer/proto/framework.nb.h"

namespace paddle {
namespace lite {
namespace naive_buffer {

class VarDesc {
 public:
  VarDesc() = delete;

  explicit VarDesc(proto::VarDesc *desc) : desc_(desc) { CHECK(desc_); }

  void CopyFrom(VarDesc &var_desc) {
    CHECK(var_desc.Proto()) << "Source proto::VarDesc pointer can't be null";
    desc_ = var_desc.Proto();
  }

  proto::VarDesc *Proto() { return desc_; }

  const proto::VarDesc &ReadonlyProto() const { return *desc_; }

  std::string Name() const;

  void SetName(std::string name);

  VarDescAPI::VarDataType GetType() const;

  void SetType(VarDescAPI::VarDataType type);

  bool Persistable() const;

  void SetPersistable(bool persistable);

 private:
  const proto::VarType &GetVarType() const;
  proto::VarType *GetMutableVarType();

  proto::VarDesc *desc_;
};

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
