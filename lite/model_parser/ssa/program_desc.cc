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

#include <map>
#include <set>
#include <string>
#include <utility>

#include "lite/model_parser/ssa/program_desc.h"

namespace paddle {
namespace lite {
namespace general {
namespace ssa {

PlainProgramDesc::PlainProgramDesc(const general::ProgramDesc& program_desc)
    : src_desc_{&program_desc} {
  if (program_desc.HasVersion()) version_ = program_desc.Version();
  blocks_.resize(src_desc_->BlocksSize());
  block_visited_.resize(src_desc_->BlocksSize());
  InitBlocks();
  InsertOpOfBlocks();
}

void PlainProgramDesc::InitBlock(const general::BlockDesc& current,
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
    const auto* op_desc = current.GetOp<general::OpDesc>(i);
    if (BlockOpGen::instance().IsBlockOp(op_desc->Type())) {
      int sub_block_idx{op_desc->GetAttr<int>(BlockOpProtoRegistry::instance()
                                                  .GetProto(op_desc->Type())
                                                  ->AttrKey())};
      InitBlock(*(src_desc_->GetBlock<general::BlockDesc>(sub_block_idx)),
                &current);
    }
  }
}

void PlainProgramDesc::InitBlocks() {
  std::fill(block_visited_.begin(), block_visited_.end(), false);
  InitBlock(*(src_desc_->GetBlock<general::BlockDesc>(0)), nullptr);
}

void PlainProgramDesc::InsertOpOfBlock(const general::BlockDesc& block_desc) {
  int32_t block_idx{block_desc.Idx()};
  CHECK_LT(block_idx, static_cast<int>(block_visited_.size()));
  CHECK(!block_visited_[block_idx]);
  block_visited_[block_idx] = true;
  for (size_t i = 0; i < block_desc.OpsSize(); ++i) {
    const auto* raw_op = block_desc.GetOp<general::OpDesc>(i);
    auto& dst_block = blocks_[block_idx];
    if (BlockOpGen::instance().IsBlockOp(raw_op->Type())) {
      std::unique_ptr<BlockOpDesc> op{BlockOpGen::instance().NewOp(
          *raw_op, *(dst_block->scope()), block_idx)};
      auto sub_id = raw_op->GetAttr<int>(op->proto().lock()->AttrKey());
      const auto& raw_sub = *(src_desc_->GetBlock<general::BlockDesc>(sub_id));
      InsertOpOfBlock(raw_sub);
      op->UpdateInputOutput(*raw_op, *(dst_block->scope()));
      blocks_[sub_id]->SetBlockOpDesc(op.get());
      blocks_[sub_id]->AddBlockInputs(
          op->extra_inputs().cbegin(), op->extra_inputs().cend(), true);
      const auto& inputs = ConvertToSet(op->inputs());
      const auto& outputs = ConvertToSet(op->outputs());
      blocks_[sub_id]->AddBlockInputs(inputs.cbegin(), inputs.cend());
      blocks_[sub_id]->AddBlockOutputs(outputs.cbegin(), outputs.cend());
      dst_block->AddOp(std::move(op));
    } else {
      std::unique_ptr<OpDescBase> op;
      if (raw_op->Type() == "write_to_array") {
        op.reset(
            new WriteToArrayOpDesc(*raw_op, *dst_block->scope(), block_idx));
      } else if (raw_op->Type() == "read_from_array") {
        op.reset(
            new ReadFromArrayOpDesc(*raw_op, *dst_block->scope(), block_idx));
      } else {
        op.reset(new OpDesc(*raw_op, *dst_block->scope(), block_idx));
      }
      const auto& inputs = ConvertToSet(op->inputs());
      const auto& outputs = ConvertToSet(op->outputs());
      dst_block->AddOp(std::move(op));
      dst_block->AddBlockInputs(inputs.cbegin(), inputs.cend());
      dst_block->AddBlockOutputs(outputs.cbegin(), outputs.cend());
    }
  }
}

void PlainProgramDesc::InsertWriteBackOp(
    const std::unique_ptr<BlockDesc>& block) {
  std::map<std::weak_ptr<VarDesc>,
           std::pair<std::weak_ptr<VarDesc>, std::weak_ptr<VarDesc>>,
           VarDescLT>
      clusters;
  std::set<std::weak_ptr<VarDesc>, VarDescLT> block_all_inputs;

  std::merge(block->block_inputs().cbegin(),
             block->block_inputs().cend(),
             block->block_extra_inputs().cbegin(),
             block->block_extra_inputs().cend(),
             std::inserter(block_all_inputs, block_all_inputs.begin()),
             VarDescLT());
  for (auto& input : block_all_inputs) {
    auto root = block->scope()->GetRootVarDesc(input.lock()->root_name());
    if (clusters.find(root) == clusters.end() ||
        *input.lock() < *clusters[root].first.lock()) {
      if (input.lock()->block_idx() != block->idx()) {
        clusters[root] = {input, {}};
      }
    }
  }
  for (auto& output : block->block_outputs()) {
    auto root = block->scope()->GetRootVarDesc(output.lock()->root_name());
    if (clusters.find(root) != clusters.end() &&
        output.lock() != clusters[root].first.lock()) {
      if (clusters[root].second.expired() ||
          !(*output.lock() < *clusters[root].first.lock())) {
        clusters[root].second = output;
      }
    }
  }
  for (auto& elem : clusters) {
    auto& pair = elem.second;
    if (!pair.first.expired() && !pair.second.expired()) {
      pair.second.lock()->ResetBlockIdx(pair.first.lock()->block_idx());
      if (pair.second.lock()->GetType() == VarDataType::LOD_TENSOR_ARRAY) {
        std::unique_ptr<OpDescBase> op{
            new WriteBackOp{pair.second, pair.first, block->idx(), true}};
        auto input_lod_deps =
            static_cast<WriteBackOp*>(op.get())->input_lod_deps();
        op->set_tensor_array_copy();
        block->AddOp(std::move(op));
        block->AddBlockInputs(&pair.first, &pair.first + 1);
        block->AddBlockInputs(input_lod_deps.begin(), input_lod_deps.end());
      } else {
        std::unique_ptr<OpDescBase> op{
            new WriteBackOp{pair.second, pair.first, block->idx(), false}};
        auto input_lod_deps =
            static_cast<WriteBackOp*>(op.get())->input_lod_deps();
        block->AddOp(std::move(op));
        block->AddBlockInputs(&pair.first, &pair.first + 1);
        block->AddBlockInputs(input_lod_deps.begin(), input_lod_deps.end());
      }
    }
  }
}

void PlainProgramDesc::UpdateBlockOp(const std::unique_ptr<BlockDesc>& block) {
  auto* block_op = block->mutable_block_op();
  if (block_op) {
    for (auto& input : block->block_inputs()) {
      if (input.lock()->block_idx() != block->idx()) {
        // In a block that has undergone correct static single assignment
        // processing, if a variable is input and output at the same time,
        // then it must be the parent block variable that is assigned for
        // the first time in the child block. To avoid loops in this case,
        // just treat the variable as an output.
        //
        // TensorArray is an exception, it allows multiple writes, so its
        // dependencies are limited by redundant variables, and the redundant
        // inputs are deleted.
        if (block->block_outputs().find(input) ==
            block->block_outputs().end()) {
          block_op->AddBlockInput(input);
        }
      }
    }
    for (auto& output : block->block_outputs()) {
      if (output.lock()->block_idx() != block->idx()) {
        block_op->AddBlockOutput(output);
      }
    }
  }
}

void PlainProgramDesc::InsertOpOfBlocks() {
  std::fill(block_visited_.begin(), block_visited_.end(), false);
  InsertOpOfBlock(*(src_desc_->GetBlock<general::BlockDesc>(0)));
  for (size_t i = 0; i < block_visited_.size(); ++i) {
    if (!block_visited_[i]) {
      LOG(WARNING) << "The block " << i << " fill error.";
    }
  }
  for (const auto& block : blocks_) {
    CHECK(block);
    if (block->parent()) {
      InsertWriteBackOp(block);
    }
    UpdateBlockOp(block);
  }
}

ProgramDescConverter::ProgramDescConverter(const PlainProgramDesc& program_desc)
    : src_desc_{&program_desc} {
  desc_.SetVersion(program_desc.Version());
  InitBlocks();
}

void ProgramDescConverter::InitBlocks() {
  for (auto& block : src_desc_->blocks()) {
    auto* dst_block = desc_.AddBlock<general::BlockDesc>();
    dst_block->SetIdx(block->idx());
    dst_block->SetParentIdx(0);
    dst_block->SetForwardBlockIdx(0);
    if (block->parent()) {
      dst_block->SetParentIdx(block->parent()->idx());
    }
    if (block->kids().size()) {
      dst_block->SetForwardBlockIdx(block->kids().front()->idx());
    }
  }
  for (auto& block : src_desc_->blocks()) {
    InitBlockOps(*block);
    InitVars(*block);
  }
}

void ProgramDescConverter::SetVar(const VarDesc& var) {
  CHECK_GE(var.block_idx(), 0);
  CHECK_LT(var.block_idx(), static_cast<int32_t>(src_desc_->blocks().size()));
  auto* block = desc_.GetBlock<general::BlockDesc>(var.block_idx());
  auto* dst_var = block->AddVar<general::VarDesc>();
  *dst_var = *var.root_var_desc();
  dst_var->SetName(var.mangled_name());
}

void ProgramDescConverter::InitVars(const BlockDesc& src_block) {
  for (auto& src_root_var : src_block.scope()->GetRootVars()) {
    SetVar(*src_root_var.lock());
    for (const auto& kid : src_root_var.lock()->series()) {
      SetVar(*kid.lock());
    }
  }
}

void ProgramDescConverter::InitBlockOps(const BlockDesc& src_block) {
  auto* dst_block = desc_.GetBlock<general::BlockDesc>(src_block.idx());
  for (auto& src_op : src_block.ops()) {
    auto* dst_op = dst_block->AddOp<general::OpDesc>();
    *dst_op = src_op->src_raw_desc();
    if (src_op->tensor_array_copy()) {
      dst_op->SetAttr<bool>("tensor_array_copy", true);
    }
    for (auto& input : src_op->inputs()) {
      std::vector<std::string> args;
      for (auto& var : input.second) {
        args.emplace_back(var.lock()->mangled_name());
      }
      dst_op->SetInput(input.first, std::move(args));
    }
    for (auto& output : src_op->outputs()) {
      std::vector<std::string> args;
      for (auto& var : output.second) {
        args.emplace_back(var.lock()->mangled_name());
      }
      dst_op->SetOutput(output.first, std::move(args));
    }
  }
}

void ConvertToSSA(general::ProgramDesc* prog) {
  general::ssa::PlainProgramDesc plain_prog(*prog);
  general::ssa::ProgramDescConverter prog_converter(plain_prog);
  *prog = prog_converter.general_program();
}

}  // namespace ssa
}  // namespace general
}  // namespace lite
}  // namespace paddle
