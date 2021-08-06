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

#include "lite/model_parser/general/program_desc.h"
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

// Algorithm Description
//
// 1. Record the input and output relationship between the current operator
//    block variables and the operator.
// 2. When the same variable is input as an operator and then output as
//    another operator, rename it to a new variable. The new name is
//    composed of the original name plus an increasing number. The specific
//    rules are as follows: if the operator is a simple operator, the
//    number will be incremented by one. If it is an internal operator block,
//    it will enter the operator block recursively, and process step 1 - 3.
//    Use the maximum increment value of the number as the renamed value of
//    the output variable of the internal operator block. All the original
//    name variables that appear after the external operator block point to
//    the new name. Finally, the write-back variable and the write-back
//    target variable are placed in the same block. The new variable in the
//    middle of the block and the operators connected to both sides of it
//    are placed in the same block.
// 3. Record the time when the original variable was output as an operator
//    for the last time, according to whether to add a write-back operator
//    in the loop block, and use it as an input between this time and the
//    previous time as an output as a dependency.
//
//
// Notice
//
// (I) Due to algorithm limitations, the directed ring cannot be removed
//     in the two topologies.
//
// 1. When the WriteToArray operator repeatedly writes different Tensor to
//    the same index, data competition will occur at this time.
// 2. When the WriteToArray operator is followed by ReadFromArray, and
//    then another WriteToArray operator is followed again by the first
//    WriteToArray operator, and the same output is written to WriteToArray
//    twice, there will still be a directed loop in the calculation graph at
//    this time.
// 3. The above two limitations are due to the fact that the main framework
//    is too simple to abstract TensorArray; thus, the input of the
//    WriteToArray operator is not equivalent to its data dependence. Almost
//    no one will trigger the above two extreme situations during networking,
//    so no further processing will be done. To deal with other problems
//    caused by WriteToArray, an additional joint output AssociatedOut is
//    added to indicate data dependencies.
//
// (II) There are two boundary cases that satisfy directed loop, but since
//      the rest of the frame has not been modified yet, it cannot complete
//      calculation.
//
// 1. There is a While operator in the network, but the while condition
//    variable is False for the first reasoning (there are several solutions
//    to this problem, to be developed).
// 2. When different operators write to the same LoDTensor multiple times,
//    and the write destination device type requirements are different
//    (complement the operator implementation of the specified device to avoid
//    this situation).

void ConvertToSSA(general::ProgramDesc* prog);

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
