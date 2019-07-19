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

#include <vector>
#include "lite/model_parser/desc_apis.h"
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
