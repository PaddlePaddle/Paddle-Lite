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

#include <map>
#include <string>
#include <vector>
#include "lite/core/model/base/apis.h"
#include "lite/model_parser/naive_buffer/proto/framework.nb.h"

namespace paddle {
namespace lite {
namespace naive_buffer {

class ProgramDesc : public ProgramDescAPI {
 public:
  ProgramDesc() = delete;

  explicit ProgramDesc(proto::ProgramDesc *desc) : desc_(desc) { CHECK(desc_); }

  void CopyFrom(ProgramDesc &program_desc) {
    CHECK(program_desc.Proto())
        << "Source proto::ProgramDesc pointer can't be null";
    desc_ = program_desc.Proto();
  }

  proto::ProgramDesc *Proto() { return desc_; }

  const proto::ProgramDesc &ReadonlyProto() const { return *desc_; }

  size_t BlocksSize() const override;

  void ClearBlocks() override;

  template <typename T>
  T *GetBlock(int32_t idx);

  template <typename T>
  T *AddBlock();

  /////////////////////////////////////////////////////////////////
  // Name: OpVersionMap
  // Description: a map that strores paddle ops version
  /////////////////////////////////////////////////////////////////

  // note: naive_buffer doesn't contain op_version_map, because
  //       op_version_map is not useful in inference period.
  bool HasOpVersionMap() const override { return false; }

  template <typename T>
  T *GetOpVersionMap();

  void SetOpVersionMap(std::map<std::string, int32_t> op_version_map) {}

  bool HasVersion() const override { return true; }

  int64_t Version() const override;

  void SetVersion(int64_t version) override;

 private:
  const ListBuilder<proto::BlockDesc> &GetBlockListBuilder() const;
  ListBuilder<proto::BlockDesc> *GetMutableBlockListBuilder();

  proto::ProgramDesc *desc_;
};

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
