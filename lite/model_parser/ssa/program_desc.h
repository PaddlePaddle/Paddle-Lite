// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <vector>

#include "lite/core/model/general/program_desc.h"
#include "lite/model_parser/ssa/block_desc.h"

namespace paddle {
namespace lite {
namespace general {
namespace ssa {

// The original program desc used the Operator to recursively include
// the block desc data structure, which is not conducive to the algorithm
// to sort out the dependencies between variables, especially inconvenient
// to express the block reference relationship, so the plain program desc
// is used here.

class PlainProgramDesc {
 public:
  // The plain program desc is constructed by the general program desc,
  // and we take the plain program desc as the intermediate representation
  // for directed loop processing.
  explicit PlainProgramDesc(const general::ProgramDesc& program_desc);

  const std::vector<std::unique_ptr<BlockDesc>>& blocks() const {
    return blocks_;
  }

  int64_t Version() const { return version_; }

 protected:
  void InitBlock(const general::BlockDesc& current,
                 const general::BlockDesc* parent);
  void InitBlocks();
  void InsertOpOfBlock(const general::BlockDesc& block_desc);
  void InsertWriteBackOp(const std::unique_ptr<BlockDesc>& block);
  void UpdateBlockOp(const std::unique_ptr<BlockDesc>& block);
  void InsertOpOfBlocks();

 private:
  std::vector<std::unique_ptr<BlockDesc>> blocks_;
  const general::ProgramDesc* src_desc_{nullptr};
  std::vector<bool> block_visited_;
  int64_t version_{0};
};

// Convert plain program desc back to normal form.
class ProgramDescConverter {
 public:
  explicit ProgramDescConverter(const PlainProgramDesc& program_desc);
  const general::ProgramDesc& general_program() const { return desc_; }

 protected:
  void InitBlocks();
  void SetVar(const VarDesc& var);
  void InitVars(const BlockDesc& src_block);
  void InitBlockOps(const BlockDesc& src_block);

 private:
  general::ProgramDesc desc_;
  const PlainProgramDesc* src_desc_;
};

void ConvertToSSA(general::ProgramDesc* prog);

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
