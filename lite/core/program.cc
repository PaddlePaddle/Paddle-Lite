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
    LOG(INFO) << ">> Running kernel: " << inst.op()->op_info()->Repr()
              << " on Target " << TargetToStr(inst.kernel()->target());

    inst.Run();
    auto tensor_mean = [](const Tensor* in, PrecisionType ptype) -> double {
      double sum = 0.;
      switch (ptype) {
        case PRECISION(kFloat): {
          auto ptr = in->data<float>();
          if (!ptr) {
            return 0;
          }
          for (int i = 0; i < in->numel(); ++i) {
            sum += ptr[i];
          }
          return sum / in->numel();
        }
        case PRECISION(kInt8): {
          auto ptr = in->data<int8_t>();
          for (int i = 0; i < in->numel(); ++i) {
            sum += ptr[i];
          }
          return sum / in->numel();
        }
        case PRECISION(kInt32): {
          auto ptr = in->data<int32_t>();
          for (int i = 0; i < in->numel(); ++i) {
            sum += ptr[i];
          }
          return sum / in->numel();
        }
        default:
          LOG(INFO) << "unsupport data type: " << PrecisionToStr(ptype);
          return 0.;
      }
    };
    if (inst.op()->op_info()->Type() != "fetch" &&
        inst.op()->op_info()->Type() != "write_to_array" &&
        inst.op()->op_info()->Type() != "while") {
      auto op = const_cast<lite::OpLite*>(inst.op());
      auto kernel = inst.kernel();
      auto op_scope = op->scope();
      auto out_names = op->op_info()->output_names();
      for (auto& out_name : out_names) {
        std::string out_arg_name;
        op->op_info()->GetOutputArgname(out_name, &out_arg_name);
        auto type = kernel->GetOutputDeclType(out_arg_name);
        if (type->IsTensor()) {
          auto tout = op_scope->FindVar(out_name)->GetMutable<Tensor>();
          double mean = tensor_mean(tout, type->precision());
          LOG(INFO) << "output name: " << out_name << ", dims: " << tout->dims()
                    << ", precision: " << PrecisionToStr(type->precision())
                    << ", mean value: " << mean;
        } else if (type->IsTensorList()) {
          auto tout =
              op_scope->FindVar(out_name)->GetMutable<std::vector<Tensor>>();
          for (auto& t : *tout) {
            double mean = tensor_mean(&t, type->precision());
            LOG(INFO) << "output name: " << out_name << ", dims: " << t.dims()
                      << ", precision: " << PrecisionToStr(type->precision())
                      << ", mean value: " << mean;
          }
        }
      }
    }

    LOG(INFO) << inst.op()->op_info()->Repr() << "finished";
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
      auto sub_block =
          const_cast<cpp::ProgramDesc&>(prog).GetBlock<cpp::BlockDesc>(
              sub_block_idx);
      static_cast<operators::WhileOpLite*>(op.get())->SetSubBlock(sub_block);
    }
    ops_.emplace_back(std::move(op));
    ops_.back()->Attach(op_desc, exec_scope_);
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
  for (size_t b = 0; b < program.BlocksSize(); ++b) {
    auto& main_block = *program.GetBlock<cpp::BlockDesc>(b);
    for (size_t i = 0; i < main_block.VarsSize(); ++i) {
      auto& var_desc = *main_block.GetVar<cpp::VarDesc>(i);
      if (!var_desc.Persistable()) {
        tmp_vars_.push_back(var_desc.Name());
        exec_scope_->Var(var_desc.Name());
        if (b > 0) {
          LOG(INFO) << "var: " << var_desc.Name();
        }
      } else {
        if (var_desc.Name() == "feed" || var_desc.Name() == "fetch") continue;
        weights_.push_back(var_desc.Name());
        if (var_desc.Persistable()) scope_->Var(var_desc.Name());
      }
    }
  }
}

void Instruction::Run() {
#ifdef LITE_WITH_PROFILE
  profile::ProfileBlock x(profile_id_);
#endif  // LITE_WITH_PROFILE
  CHECK(op_) << "op null";
  CHECK(kernel_) << "kernel null";
  if (first_epoch_) {
    first_epoch_ = false;
    CHECK(op_->CheckShape());
  }

  if (op_->run_once() && has_run_) return;
  LOG(INFO) << "op infershape";
  op_->InferShape();
  LOG(INFO) << "kernel launch";
  kernel_->Launch();
  LOG(INFO) << "kernel launched";
  has_run_ = true;
}

std::ostream& operator<<(std::ostream& os, const Instruction& other) {
  os << other.kernel_->summary() << "\t(" << other.kernel_->doc() << ")";
  return os;
}

}  // namespace lite
}  // namespace paddle
