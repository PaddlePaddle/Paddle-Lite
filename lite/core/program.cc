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
#include <algorithm>
#include <unordered_map>
#include "lite/model_parser/cpp/block_desc.h"
#include "lite/model_parser/cpp/op_desc.h"
#include "lite/model_parser/cpp/var_desc.h"
#include "lite/operators/conditional_block_op.h"
#include "lite/operators/subgraph_op.h"
#include "lite/operators/while_op.h"
#ifdef LITE_WITH_PRECISION_PROFILE
#include "lite/core/profile/precision_profiler.h"
#endif

namespace paddle {
namespace lite {

void RuntimeProgram::SaveOpInfosToProgram(cpp::ProgramDesc* desc) {
  CHECK(desc);
  // NOTE: RuntimeProgram do not has all meta info, so save model just update
  // upon origin model
  CHECK(desc->BlocksSize());
  auto main_block = desc->GetBlock<cpp::BlockDesc>(0);
  main_block->ClearOps();
  for (auto& node : instructions_) {
    auto op_type = node.op()->op_info()->Type();
    if (op_type == "subgraph") {
      auto subgraph_op = const_cast<operators::SubgraphOp*>(
          static_cast<const operators::SubgraphOp*>(node.op()));
      int sub_block_idx = subgraph_op->op_info()->GetAttr<int32_t>("sub_block");
      if (sub_block_idx < 0) {
        // It's a new subgraph op when its sub_block_idx < 0, Now we add its
        // subblock desc to the program desc, Then update its sub_block_idx to
        // the index of block desc of the program desc.
        sub_block_idx = desc->BlocksSize();
        auto sub_block_desc = subgraph_op->GetSubBlock();
        CHECK(sub_block_desc);
        auto new_block_desc = desc->AddBlock<cpp::BlockDesc>();
        *new_block_desc = *sub_block_desc;
        delete sub_block_desc;
        subgraph_op->mutable_op_info()->SetAttr<int32_t>("sub_block",
                                                         sub_block_idx);
        subgraph_op->SetSubBlock(new_block_desc);
        // Update main block desc after a new subblock desc is added
        main_block = desc->GetBlock<cpp::BlockDesc>(0);
      }
    }
    auto op = main_block->AddOp<cpp::OpDesc>();
    *op = *node.op()->op_info();
    op->SetAttr(kKernelTypeAttr, node.kernel()->SerializedKernelType());
  }
}

// `UpdateVarsOfProgram` will remove unused var_descs and add new created
// vars' descs in the block 0. Now, the type of a new created var can only
// be LOD_TENSOR.
void RuntimeProgram::UpdateVarsOfProgram(cpp::ProgramDesc* desc) {
  CHECK(desc);
  CHECK(desc->BlocksSize());
  std::unordered_map<std::string, cpp::VarDesc> origin_var_maps;
  auto& main_block = *desc->GetBlock<cpp::BlockDesc>(0);
  auto var_size = main_block.VarsSize();
  for (int i = 0; i < var_size; i++) {
    auto v = main_block.GetVar<cpp::VarDesc>(i);
    auto name = v->Name();
    origin_var_maps.emplace(name, *v);
  }

  main_block.ClearVars();
  for (auto& node : instructions_) {
    auto* op = const_cast<lite::OpLite*>(node.op());
    auto* kernel = node.kernel();
    auto* scope = op->scope();
    auto in_names = op->op_info()->input_names();
    auto out_names = op->op_info()->output_names();
    in_names.insert(in_names.end(), out_names.begin(), out_names.end());
    std::sort(in_names.begin(), in_names.end());
    in_names.erase(std::unique(in_names.begin(), in_names.end()),
                   in_names.end());
    for (auto& in_name : in_names) {
      auto it = origin_var_maps.find(in_name);
      if (it != origin_var_maps.end()) {
        auto* v = main_block.AddVar<cpp::VarDesc>();
        v->SetName((it->second).Name());
        v->SetType((it->second).GetType());
        v->SetPersistable((it->second).Persistable());
        if ((it->second).Name() != "feed" && (it->second).Name() != "fetch") {
          v->SetShape((it->second).GetShape());
          v->SetDataType((it->second).GetDataType());
        }
      } else {
        // New created vars must be LOD_TENSOR
        auto* v = main_block.AddVar<cpp::VarDesc>();
        v->SetName(in_name);
        v->SetType(cpp::VarDesc::Type::LOD_TENSOR);
        std::string in_arg_name;
        const Type* type;
        if (op->op_info()->GetInputArgname(in_name, &in_arg_name)) {
          type = kernel->GetInputDeclType(in_arg_name);
        } else {
          op->op_info()->GetOutputArgname(in_name, &in_arg_name);
          type = kernel->GetOutputDeclType(in_arg_name);
        }
        if (type->IsTensor()) {
          auto tensor = scope->FindVar(in_name)->GetMutable<Tensor>();
          v->SetPersistable(tensor->persistable());
          if (in_name != "feed" && in_name != "fetch") {
            v->SetShape(tensor->dims().data());
            switch (tensor->precision()) {
#define SET_DATATYPE(precision__, data_type)                    \
  case PrecisionType::precision__:                              \
    v->SetDataType(data_type);                                  \
    LOG(INFO) << "update var" << (it->second).Name() << "done"; \
    break
              SET_DATATYPE(kBool, VarDescAPI::VarDataType::BOOL);
              SET_DATATYPE(kFloat, VarDescAPI::VarDataType::FP32);
              SET_DATATYPE(kFP16, VarDescAPI::VarDataType::FP16);
              SET_DATATYPE(kInt8, VarDescAPI::VarDataType::INT8);
              SET_DATATYPE(kInt16, VarDescAPI::VarDataType::INT16);
              SET_DATATYPE(kInt32, VarDescAPI::VarDataType::INT32);
              SET_DATATYPE(kInt64, VarDescAPI::VarDataType::INT64);
#undef SET_DATATYPE
              default:
                VLOG(4) << "warning! unknown precision type";
            }
          }
        } else {
          CHECK(false) << "unsupported var type";
        }
      }
    }
  }
}
void RuntimeProgram::Run() {
#ifdef LITE_WITH_PRECISION_PROFILE
  auto inst_precision_profiler = paddle::lite::profile::PrecisionProfiler();
  std::string precision_profiler_summary =
      inst_precision_profiler.GetSummaryHeader();
#endif

  for (auto& inst : instructions_) {
#ifndef LITE_WITH_FPGA
    if (inst.is_feed_fetch_op()) continue;
#endif
#ifdef LITE_WITH_CUDA
    if (inst.need_sync()) {
      inst.Sync();
    }
#endif
    inst.Run();
#ifdef LITE_WITH_PRECISION_PROFILE
#ifndef LITE_WITH_FPGA
    precision_profiler_summary +=
        inst_precision_profiler.GetInstPrecision(&inst);
#endif
#endif  // LITE_WITH_PRECISION_PROFILE
  }
#ifdef LITE_WITH_PROFILE
  LOG(INFO) << "\n" << profiler_.Summary(profile::Type::kDispatch, false, 0);
#endif
#ifdef LITE_WITH_PRECISION_PROFILE
  LOG(INFO) << "\n" << precision_profiler_summary;
#endif
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
    auto op = LiteOpRegistry::Global().Create(op_type);
    CHECK(op) << "no Op found for " << op_type;
    if (op_type == "while" || op_type == "conditional_block" ||
        op_type == "subgraph") {
      auto sub_block_idx = op_desc.GetAttr<int32_t>("sub_block");
      CHECK(sub_block_idx >= 0 && sub_block_idx < program.BlocksSize())
          << "Invalid attribute sub_block(" << sub_block_idx << ") for "
          << op_type;
      auto sub_block_desc =
          const_cast<cpp::ProgramDesc&>(prog).GetBlock<cpp::BlockDesc>(
              sub_block_idx);
      CHECK(sub_block_desc);
      if (op_type == "while") {
        static_cast<operators::WhileOpLite*>(op.get())->SetSubBlock(
            sub_block_desc);
      } else if (op_type == "conditional_block") {
        static_cast<operators::ConditionalBlockOpLite*>(op.get())->SetSubBlock(
            sub_block_desc);
      } else if (op_type == "subgraph") {
        static_cast<operators::SubgraphOp*>(op.get())->SetSubBlock(
            sub_block_desc);
      }
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

  auto VarPrecision2KernlPrecision =
      [](const lite::VarDescAPI::Type& type) -> PrecisionType {
    switch (type) {
      case lite::VarDescAPI::Type::FP32:
        return PRECISION(kFloat);
      case lite::VarDescAPI::Type::FP16:
        return PRECISION(kFP16);
      case lite::VarDescAPI::Type::INT8:
        return PRECISION(kInt8);
      case lite::VarDescAPI::Type::INT16:
        return PRECISION(kInt16);
      case lite::VarDescAPI::Type::INT32:
        return PRECISION(kInt32);
      case lite::VarDescAPI::Type::INT64:
        return PRECISION(kInt64);
      default:
        // LOG(FATAL) << "not supported type: " << static_cast<int>(type);
        return PRECISION(kUnk);
    }
  };

  auto program = prog;
  CHECK(program.BlocksSize());
  for (size_t b = 0; b < program.BlocksSize(); ++b) {
    auto& main_block = *program.GetBlock<cpp::BlockDesc>(b);
    for (size_t i = 0; i < main_block.VarsSize(); ++i) {
      auto& var_desc = *main_block.GetVar<cpp::VarDesc>(i);
      if (!var_desc.Persistable()) {
        if (var_desc.GetType() == lite::VarDescAPI::Type::LOD_TENSOR &&
            VarPrecision2KernlPrecision(var_desc.GetDataType()) !=
                PRECISION(kUnk)) {
          var_data_type_[var_desc.Name()] =
              VarPrecision2KernlPrecision(var_desc.GetDataType());
        }
        tmp_vars_.push_back(var_desc.Name());
        VLOG(4) << "var name: " << var_desc.Name() << " type is "
                << static_cast<int>(var_desc.GetType()) << " data type is "
                << static_cast<int>(var_desc.GetDataType());
        exec_scope_->Var(var_desc.Name());
        if (b > 0) {
          VLOG(4) << "var: " << var_desc.Name();
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
  CHECK(profiler_) << "Profiler pointer of kernel can not be nullptr. "
                      "When LITE_WITH_PROFILE is defined, please set a "
                      "Profiler for Instruction.";
  profiler_->StartTiming(
      profile::Type::kCreate, profile_id_, kernel_->mutable_context());
#endif
  CHECK(op_) << "op null";
  CHECK(kernel_) << "kernel null";

  if (first_epoch_) {
    first_epoch_ = false;
    CHECK(op_->CheckShape());
  }

  if (op_->run_once() && has_run_) {
    return;
  }

  op_->InferShape();
  kernel_->Launch();
  has_run_ = true;
}

STL::ostream& operator<<(STL::ostream& os, const Instruction& other) {
  os << other.kernel_->summary() << "\t(" << other.kernel_->doc() << ")";
  return os;
}

}  // namespace lite
}  // namespace paddle
