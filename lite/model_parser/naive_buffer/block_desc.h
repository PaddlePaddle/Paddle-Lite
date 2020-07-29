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

class BlockDesc : public BlockDescAPI {
 public:
  BlockDesc() = delete;

  explicit BlockDesc(proto::BlockDesc* desc) : desc_(desc) { CHECK(desc_); }

  void CopyFrom(BlockDesc& block_desc) {
    CHECK(block_desc.Proto())
        << "Source proto::BlockDesc pointer can't be null";
    desc_ = block_desc.Proto();
  }

  proto::BlockDesc* Proto() { return desc_; }

  const proto::BlockDesc& ReadonlyProto() const { return *desc_; }

  int32_t Idx() const override;

  void SetIdx(int32_t idx) override;

  int32_t ParentIdx() const override;

  void SetParentIdx(int32_t idx) override;

  size_t VarsSize() const override;

  void ClearVars() override;

  template <typename T>
  T* GetVar(int32_t idx);

  template <typename T>
  T* AddVar();

  size_t OpsSize() const override;

  void ClearOps() override;

  template <typename T>
  T* GetOp(int32_t idx);

  template <typename T>
  T* AddOp();

  int32_t ForwardBlockIdx() const override;

  void SetForwardBlockIdx(int32_t idx) override;

 private:
  const ListBuilder<proto::VarDesc>& GetVarListBuilder() const;
  ListBuilder<proto::VarDesc>* GetMutableVarListBuilder();
  const ListBuilder<proto::OpDesc>& GetOpListBuilder() const;
  ListBuilder<proto::OpDesc>* GetMutableOpListBuilder();

  proto::BlockDesc* desc_;
};

}  // namespace naive_buffer
}  // namespace lite
}  // namespace paddle
