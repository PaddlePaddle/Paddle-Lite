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
#include "lite/model_parser/base/apis.h"
#include "lite/model_parser/naive_buffer/proto/framework.nb.h"

namespace paddle {
namespace lite {
namespace naive_buffer {

class ParamDesc {
 public:
  ParamDesc() = delete;

  explicit ParamDesc(proto::ParamDesc *desc) : desc_(desc) { CHECK(desc_); }

  void CopyFrom(ParamDesc &param_desc) {
    CHECK(param_desc.Proto())
        << "Source proto::ParamDesc pointer can't be null";
    desc_ = param_desc.Proto();
  }

  proto::ParamDesc *Proto() { return desc_; }

  const proto::ParamDesc &ReadonlyProto() const { return *desc_; }

  std::string Name() const;

  void SetName(const std::string &name);

  uint32_t ModelVersion() const;

  void SetModelVersion(uint32_t version);

  uint32_t TensorVersion() const;

  void SetTensorVersion(uint32_t version);

  uint64_t LoDLevel() const;

  void SetLoDLevel(uint64_t lod_level);

  std::vector<std::vector<uint64_t>> LoD() const;

  void SetLoD(const std::vector<std::vector<uint64_t>> &lod);

  VarDescAPI::VarDataType GetDataType() const;

  void SetDataType(VarDescAPI::VarDataType data_type);

  std::vector<int64_t> Dim() const;

  void SetDim(const std::vector<int64_t> &dim);

  template <typename T>
  std::vector<T> Data() const;

  template <typename T>
  void SetData(const std::vector<T> &data);

  template <typename T>
  void SetData(const T *data, size_t size);

 private:
  uint32_t Version(const std::string &name) const;
  void SetVersion(const std::string &name, uint32_t version);

  const proto::TensorDesc &GetTensorDesc() const;
  proto::TensorDesc *GetMutableTensorDesc();

  proto::ParamDesc *desc_;
};

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
