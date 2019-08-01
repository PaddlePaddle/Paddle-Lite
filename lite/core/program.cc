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

#include "lite/core/program.h"
#include "lite/model_parser/cpp/block_desc.h"
#include "lite/model_parser/cpp/op_desc.h"
#include "lite/model_parser/cpp/var_desc.h"
#include "lite/operators/while_op.h"

namespace paddle {
namespace lite {

void RuntimeProgram::SaveOpInfosToProgram(cpp::ProgramDesc* desc) {
  CHECK(desc);
  // NOTE: RuntimeProgram do not has all meta info, so save model just update
  // upon origin model
  CHECK(desc->BlocksSize());
  auto& main_block = *desc->GetBlock<cpp::BlockDesc>(0);
  main_block.ClearOps();
  for (auto& node : instructions_) {
    auto* op = main_block.AddOp<cpp::OpDesc>();
    *op = *node.op()->op_info();
    op->SetAttr(kKernelTypeAttr, node.kernel()->SerializedKernelType());
  }
}

void RuntimeProgram::Run() {
  for (auto& inst : instructions_) {
    VLOG(4) << ">> Running kernel: " << inst.op()->op_info()->Repr()
            << " on Target " << TargetToStr(inst.kernel()->target());
    inst.Run();
  }
}

void Program::Build(const cpp::ProgramDesc& prog) {
  CHECK(ops_.empty()) << "Executor duplicate Build found";

  // Create operators.
  auto program = prog;
  CHECK(program.BlocksSize());
  auto& main_block = *program.GetBlock<cpp::BlockDesc>(0);
  for (size_t i = 0; i < main_block.OpsSize(); ++i) {
    auto& op_desc = *main_block.GetOp<cpp::OpDesc>(i);
    auto op_type = op_desc.Type();
    // if (op_type == "feed" || op_type == "fetch") continue;
    VLOG(4) << "create Op [" << op_type << "]";
    LOG(INFO) << "create Op [" << op_type << "]";
    auto op = LiteOpRegistry::Global().Create(op_type);
    CHECK(op) << "no Op found for " << op_type;
    if (op_type == "while") {
      auto sub_block_idx = op_desc.GetAttr<int16_t>("sub_block");
      LOG(INFO) << sub_block_idx;
      auto sub_block = program.GetBlock<cpp::BlockDesc>(sub_block_idx);
      LOG(INFO) << sub_block_idx;
      static_cast<operators::WhileOpLite*>(op.get())->SetSubBlock(sub_block);
      LOG(INFO) << sub_block_idx;
    }
    ops_.emplace_back(std::move(op));
    ops_.back()->Attach(op_desc, exec_scope_);
    LOG(INFO) << "attached";
  }
}

void Program::PrepareWorkspace(const cpp::ProgramDesc& prog) {
  CHECK(!exec_scope_) << "Duplicate PrepareWorkspace found";
  exec_scope_ = &scope_->NewScope();
  // Create Feed and Fetch var.
  scope_->Var("feed")->GetMutable<std::vector<lite::Tensor>>();
  scope_->Var("fetch")->GetMutable<std::vector<lite::Tensor>>();
  tmp_vars_.push_back("feed");
  tmp_vars_.push_back("fetch");

  auto program = prog;
  CHECK(program.BlocksSize());
  auto& main_block = *program.GetBlock<cpp::BlockDesc>(0);
  for (size_t i = 0; i < main_block.VarsSize(); ++i) {
    auto& var_desc = *main_block.GetVar<cpp::VarDesc>(i);
    if (!var_desc.Persistable()) {
      tmp_vars_.push_back(var_desc.Name());
      exec_scope_->Var(var_desc.Name());
    } else {
      if (var_desc.Name() == "feed" || var_desc.Name() == "fetch") continue;
      weights_.push_back(var_desc.Name());
      if (var_desc.Persistable()) scope_->Var(var_desc.Name());
    }
  }
}

void Instruction::Run() {
#ifdef LITE_WITH_PROFILE
  profile::ProfileBlock x(profile_id_);
#endif  // LITE_WITH_PROFILE
  CHECK(op_);
  CHECK(kernel_);
  if (first_epoch_) {
    first_epoch_ = false;
    CHECK(op_->CheckShape());
  }

  if (op_->run_once() && has_run_) return;
  op_->InferShape();
  kernel_->Launch();
  has_run_ = true;
}

std::ostream& operator<<(std::ostream& os, const Instruction& other) {
  os << other.kernel_->summary() << "\t(" << other.kernel_->doc() << ")";
  return os;
}

}  // namespace lite
}  // namespace paddle
