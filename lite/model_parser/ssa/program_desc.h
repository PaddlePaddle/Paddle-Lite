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
#include <string>
#include <utility>
#include <vector>

#include "lite/model_parser/general/program_desc.h"
#include "lite/model_parser/ssa/block_desc.h"

namespace paddle {
namespace lite {
namespace general {
namespace ssa {

class PlainProgramDesc {
 public:
  explicit PlainProgramDesc(const general::ProgramDesc& program_desc)
      : src_desc_{&program_desc} {
    blocks_.resize(src_desc_->BlocksSize());
    block_visited_.resize(src_desc_->BlocksSize());
    InitBlocks();
    InsertOpOfBlocks();
  }

  const std::vector<std::unique_ptr<BlockDesc>>& blocks() const {
    return blocks_;
  }

 protected:
  void InitBlock(const general::BlockDesc& current,
                 const general::BlockDesc* parent) {
    int32_t block_idx{current.Idx()};
    CHECK(!block_visited_[block_idx]);
    block_visited_[block_idx] = true;
    if (parent) {
      blocks_[current.Idx()].reset(
          new BlockDesc{current, blocks_[parent->Idx()].get()});
    } else {
      blocks_[current.Idx()].reset(new BlockDesc{current});
    }
    for (size_t i = 0; i < current.OpsSize(); ++i) {
      const auto* op_desc{current.GetOp<general::OpDesc>(i)};
      if (BlockParamInfo::instance().IsBlockOp(op_desc->Type())) {
        int sub_block_idx{op_desc->GetAttr<int>(
            BlockParamInfo::instance().GetBlockAttrName(op_desc->Type()))};
        InitBlock(*(src_desc_->GetBlock<general::BlockDesc>(sub_block_idx)),
                  &current);
      }
    }
  }

  void InitBlocks() {
    std::fill(block_visited_.begin(), block_visited_.end(), false);
    InitBlock(*(src_desc_->GetBlock<general::BlockDesc>(0)), nullptr);
  }

  void InsertOpOfBlock(const general::BlockDesc& block_desc) {
    int32_t block_idx{block_desc.Idx()};
    CHECK_LT(block_idx, static_cast<int>(block_visited_.size()));
    CHECK(!block_visited_[block_idx]);
    block_visited_[block_idx] = true;
    for (size_t i = 0; i < block_desc.OpsSize(); ++i) {
      const auto* raw_op_desc{block_desc.GetOp<general::OpDesc>(i)};
      if (BlockParamInfo::instance().IsBlockOp(raw_op_desc->Type())) {
        int sub_block_idx{raw_op_desc->GetAttr<int>(
            BlockParamInfo::instance().GetBlockAttrName(raw_op_desc->Type()))};
        InsertOpOfBlock(
            *(src_desc_->GetBlock<general::BlockDesc>(sub_block_idx)));
      } else {
        auto op_desc{
            OpDesc{*raw_op_desc, *(blocks_[block_idx]->scope()), block_idx}};
        blocks_[block_idx]->AddOp(std::move(op_desc));
      }
    }
  }

  void InsertOpOfBlocks() {
    std::fill(block_visited_.begin(), block_visited_.end(), false);
    InsertOpOfBlock(*(src_desc_->GetBlock<general::BlockDesc>(0)));
  }

 private:
  std::vector<std::unique_ptr<BlockDesc>> blocks_;
  const general::ProgramDesc* src_desc_{nullptr};
  std::vector<bool> block_visited_;
};

class ProgramDescConverter {
 public:
  explicit ProgramDescConverter(const PlainProgramDesc& program_desc)
      : src_desc_{&program_desc} {
    desc_.SetVersion(0);
    InitBlocks();
  }

  const general::ProgramDesc& general_program() const { return desc_; }

 protected:
  void InitBlocks() {
    for (auto& block : src_desc_->blocks()) {
      auto* dst_block{desc_.AddBlock<general::BlockDesc>()};
      dst_block->SetIdx(block->idx());
      dst_block->SetParentIdx(0);
      dst_block->SetForwardBlockIdx(0);
      if (block->parent()) {
        dst_block->SetParentIdx(block->parent()->idx());
      }
      if (block->kid()) {
        dst_block->SetForwardBlockIdx(block->kid()->idx());
      }
      InitBlockOps(*block);
      InitVars(*block);
    }
  }

  void SetVar(const VarDesc& var) {
    auto* block{desc_.GetBlock<general::BlockDesc>(var.block_idx())};
    auto* dst_var{block->AddVar<general::VarDesc>()};
    *dst_var = *var.root_var_desc();
    dst_var->SetName(var.mangled_name());
  }

  void InitVars(const BlockDesc& src_block) {
    for (auto& src_root_var : src_block.scope()->GetRootVars()) {
      SetVar(*src_root_var.lock());
      for (const auto& kid : src_root_var.lock()->kids()) {
        SetVar(*kid.lock());
      }
    }
  }

  void InitBlockOps(const BlockDesc& src_block) {
    auto* dst_block{desc_.GetBlock<general::BlockDesc>(src_block.idx())};
    for (auto& src_op : src_block.ops()) {
      auto* dst_op{dst_block->AddOp<general::OpDesc>()};
      *dst_op = src_op.src_raw_desc();
      for (auto& input : src_op.inputs()) {
        std::vector<std::string> args;
        for (auto& var : input.second) {
          args.emplace_back(var.lock()->mangled_name());
        }
        dst_op->SetInput(input.first, std::move(args));
      }
      for (auto& output : src_op.outputs()) {
        std::vector<std::string> args;
        for (auto& var : output.second) {
          args.emplace_back(var.lock()->mangled_name());
        }
        dst_op->SetOutput(output.first, std::move(args));
      }
    }
  }

 private:
  general::ProgramDesc desc_;
  const PlainProgramDesc* src_desc_;
};

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
