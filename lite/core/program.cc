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
#include <map>
#include <set>
#include "lite/model_parser/cpp_desc.h"
#include "lite/operators/conditional_block_op.h"
#include "lite/operators/subgraph_op.h"
#include "lite/operators/while_op.h"
#ifdef LITE_WITH_PRECISION_PROFILE
#include "lite/core/profile/precision_profiler.h"
#endif

namespace paddle {
namespace lite {

void RuntimeProgram::SaveToProgram(
    std::shared_ptr<cpp::ProgramDesc> program_desc) {
  CHECK(program_desc);
  auto block_size = program_desc->BlocksSize();
  CHECK_GT(block_size, 0) << "No block found!";
  // TODD(hong19860320) Only support updating the block desc which already
  // exists in the origin program desc
  CHECK_LE(block_size, instructions_.size())
      << "Invalid block size, expected (0," << instructions_.size()
      << "] but got " << block_size;
  for (size_t block_idx = 0; block_idx < block_size; ++block_idx) {
    auto block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);
    // Record all of the origin vars in the origin block
    std::map<std::string, cpp::VarDesc> origin_var_maps;
    auto var_size = block_desc->VarsSize();
    for (size_t var_idx = 0; var_idx < var_size; ++var_idx) {
      auto v = block_desc->GetVar<cpp::VarDesc>(var_idx);
      origin_var_maps.emplace(v->Name(), *v);
    }
    // Update the ops and vars for each block according to the instructions
    block_desc->ClearVars();
    block_desc->ClearOps();
    std::set<std::string> already_added_vars;
    for (auto& inst : instructions_[block_idx]) {
      auto* op = const_cast<OpLite*>(inst.op());
      auto* op_info = op->op_info();
      auto op_type = op_info->Type();
      auto* kernel = inst.mutable_kernel();
      auto* scope = op->scope();
      // Update the origin vars which are referred by the instructions
      // Add the new vars which are created in the passes and referred by the
      // instructions
      auto var_names = op_info->input_names();
      auto out_names = op_info->output_names();
      // Combine input and output vars and delete the duplicates
      var_names.insert(var_names.end(), out_names.begin(), out_names.end());
      std::stable_sort(var_names.begin(), var_names.end());
      var_names.erase(std::unique(var_names.begin(), var_names.end()),
                      var_names.end());
      for (auto& var_name : var_names) {
        if (already_added_vars.count(var_name)) continue;
        auto* v = block_desc->AddVar<cpp::VarDesc>();
        v->SetName(var_name);
        auto it = origin_var_maps.find(var_name);
        if (it != origin_var_maps.end()) {
          v->SetType(it->second.GetType());
          v->SetPersistable(it->second.Persistable());
          if (var_name != "feed" && var_name != "fetch") {
            v->SetShape(it->second.GetShape());
            v->SetDataType(it->second.GetDataType());
          }
        } else {
          std::string arg_name;
          const Type* decl_type;
          if (op_info->GetInputArgname(var_name, &arg_name)) {
            decl_type = kernel->GetInputDeclType(arg_name);
          } else {
            op_info->GetOutputArgname(var_name, &arg_name);
            decl_type = kernel->GetOutputDeclType(arg_name);
          }
          if (decl_type->IsTensor()) {
            v->SetType(cpp::VarDesc::Type::LOD_TENSOR);
            auto tensor = scope->FindVar(var_name)->GetMutable<Tensor>();
            v->SetPersistable(tensor->persistable());
            if (var_name != "feed" && var_name != "fetch") {
              v->SetShape(tensor->dims().data());
              auto precision = tensor->precision();
              switch (precision) {
#define SET_DATATYPE(precision__, data_type)           \
  case PrecisionType::precision__:                     \
    v->SetDataType(data_type);                         \
    LOG(INFO) << "Update var " << var_name << " done"; \
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
                  LOG(WARNING) << "Unknown precision type "
                               << PrecisionToStr(precision) << " for var "
                               << var_name << " in op " << op_type;
              }
            }
          } else if (decl_type->IsTensorList()) {
            // Set persistable=false for tensor array
            v->SetType(cpp::VarDesc::Type::LOD_TENSOR_ARRAY);
            v->SetPersistable(false);
          } else {
            CHECK(false) << "Unsupported decl type " << *decl_type
                         << " for var " << var_name << " in op " << op_type;
          }
        }
        already_added_vars.insert(var_name);
      }
      // Replace all of origin ops with the instructions
      auto op_desc = block_desc->AddOp<cpp::OpDesc>();
      *op_desc = *op_info;
      op_desc->SetAttr(kKernelTypeAttr, kernel->SerializedKernelType());
      if (op_type == "subgraph" && !op_info->GetAttr<int32_t>("sub_block")) {
        // It's a new subgraph op when its sub_block_idx = 0, Now we add its
        // subblock desc to the program desc, Then update its sub_block_idx to
        // the index of block desc of the program desc.
        auto subgraph_op = static_cast<operators::SubgraphOp*>(op);
        auto sub_program_desc = subgraph_op->GetProgramDesc();
        CHECK(sub_program_desc);
        auto sub_block_desc = program_desc->AddBlock<cpp::BlockDesc>();
        *sub_block_desc = *sub_program_desc->GetBlock<cpp::BlockDesc>(0);
        subgraph_op->SetProgramDesc(program_desc);
        op_desc->SetAttr<int32_t>("sub_block", program_desc->BlocksSize() - 1);
        // Attach op and kernel again to update the new block_idx and
        // program_desc
        subgraph_op->Attach(*op_desc, scope);
        subgraph_op->AttachKernel(kernel);
        // Update the pointer of block desc after a new subblock desc is added
        block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);
      }
    }
  }
}

// Create runtime program from sub_block desc according to block_idx and
// program_desc, which is used for while/conditional_block/subgraph op.
RuntimeProgram::RuntimeProgram(
    const std::shared_ptr<const cpp::ProgramDesc>& program_desc,
    Scope* exec_scope,
    int block_idx)
    : exec_scope_(exec_scope) {
#ifdef LITE_WITH_OPENCL
  bool opencl_valid = CLRuntime::Global()->OpenCLAvaliableForDevice();
  using OpenCLContext = Context<TargetType::kOpenCL>;
  std::unique_ptr<KernelContext> unique_opencl_ctx(new KernelContext());
  if (opencl_valid) {
    unique_opencl_ctx->As<OpenCLContext>().InitOnce();
  }
#endif
  CHECK(program_desc);
  auto block_size = program_desc->BlocksSize();
  CHECK(block_size) << "No block found!";
  CHECK(block_idx >= 0 && block_idx < block_size)
      << "Invalid block index, expected [0," << (block_size - 1) << "] but got "
      << block_idx;
  auto block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);
  instructions_.resize(kRootBlockIdx + 1);
  auto op_size = block_desc->OpsSize();
  for (size_t op_idx = 0; op_idx < op_size; op_idx++) {
    auto op_desc = block_desc->GetOp<cpp::OpDesc>(op_idx);
    CHECK(op_desc);
    std::string op_type = op_desc->Type();
    // if (op_type == "feed" || op_type == "fetch") continue;
    // Create op and pick up the best kernel
    auto op = LiteOpRegistry::Global().Create(op_type);
    CHECK(op) << "no Op found for " << op_type;
    if (op_type == "while") {
      static_cast<operators::WhileOp*>(op.get())->SetProgramDesc(program_desc);
    } else if (op_type == "conditional_block") {
      static_cast<operators::ConditionalBlockOp*>(op.get())->SetProgramDesc(
          program_desc);
    } else if (op_type == "subgraph") {
      static_cast<operators::SubgraphOp*>(op.get())->SetProgramDesc(
          program_desc);
    }
    op->Attach(*op_desc, exec_scope_);
    std::unique_ptr<KernelBase> kernel;
    if (op_desc->HasAttr(kKernelTypeAttr)) {
      // Create op and pick up the best kernel according to the
      // kKernelTypeAttr attribute
      auto kernel_type = op_desc->GetAttr<std::string>(kKernelTypeAttr);
      std::string alias;
      Place place;
      KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
      VLOG(3) << "Found the attr '" << kKernelTypeAttr << "': " << kernel_type
              << " for " << op_type;
      auto kernels = op->CreateKernels({place});
      CHECK_GT(kernels.size(), 0) << "No kernels found for " << op_type;
      auto it = std::find_if(
          kernels.begin(), kernels.end(), [&](std::unique_ptr<KernelBase>& it) {
            return it->alias() == alias;
          });
      CHECK(it != kernels.end());
      kernel = std::move(*it);
    } else {
      // TODO(hong19860320) add kernel picking according to the type of input
      // and output tensors
      VLOG(3) << "The attr '" << kKernelTypeAttr
              << "' not found, pick the first kernel for " << op_type;
      std::vector<std::unique_ptr<KernelBase>> kernels;
#if defined(LITE_WITH_ARM)
      kernels = op->CreateKernels({Place{TARGET(kARM)}, Place{TARGET(kHost)}});
#elif defined(LITE_WITH_X86)
      kernels = op->CreateKernels({Place{TARGET(kX86)}, Place{TARGET(kHost)}});
#endif
      if (kernels.size() > 0) {
        kernel = std::move(kernels.front());
      } else {
        LOG(WARNING) << "No kernels found for " << op_type;
      }
    }
#ifdef LITE_WITH_OPENCL
    if (kernel->target() == TARGET(kOpenCL)) {
      if (opencl_valid) {
        std::unique_ptr<KernelContext> ctx(new KernelContext());
        (*unique_opencl_ctx)
            .As<OpenCLContext>()
            .CopySharedTo(&ctx->As<OpenCLContext>());
        kernel->SetContext(std::move(ctx));
      } else {
        LOG(ERROR) << "opencl_valid:" << opencl_valid;
      }
    } else {
      kernel->SetContext(
          ContextScheduler::Global().NewContext(kernel->target()));
    }
#else
    if (kernel != nullptr) {
      kernel->SetContext(
          ContextScheduler::Global().NewContext(kernel->target()));
    }
#endif
    instructions_[kRootBlockIdx].emplace_back(std::move(op), std::move(kernel));
  }
  Init();
}

void RuntimeProgram::Run() {
#ifdef LITE_WITH_PRECISION_PROFILE
  auto inst_precision_profiler = paddle::lite::profile::PrecisionProfiler();
  std::string precision_profiler_summary =
      inst_precision_profiler.GetSummaryHeader();
#endif

#ifdef LITE_WITH_NVTX
  const NVTXAnnotator& annotator = NVTXAnnotator::Global();
  NVTXRangeAnnotation annotation_one_loop = annotator.AnnotateBlock();
  if (annotator.IsEnabled()) {
    annotation_one_loop.generate(register_layer_names_.back(),
                                 lite::Color::Engine);
  }
#endif
  int idx = -1;
  auto& insts = instructions_[kRootBlockIdx];
  for (auto& inst : insts) {
    ++idx;
#ifndef LITE_WITH_FPGA
    if (inst.is_feed_fetch_op()) continue;
#endif
#ifdef LITE_WITH_NVTX
    NVTXRangeAnnotation annotation = annotator.AnnotateBlock();
    nvtxStringHandle_t registered_name = register_layer_names_[idx];
    if (annotator.IsEnabled()) {
      annotation.generate(registered_name, lite::Color::Runner);
    }
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
  LOG(INFO) << "\n" << profiler_.Summary(profile::Type::kDispatch, false, 1);
#endif
#ifdef LITE_WITH_PRECISION_PROFILE
  LOG(INFO) << "\n" << precision_profiler_summary;
#endif
}

void Program::Build(const std::shared_ptr<cpp::ProgramDesc>& program_desc) {
  CHECK(ops_.empty()) << "Executor duplicate Build found";

  // Create operators.
  auto block_size = program_desc->BlocksSize();
  CHECK(block_size);
  ops_.resize(block_size);
  for (size_t block_idx = 0; block_idx < block_size; ++block_idx) {
    auto* block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);
    auto op_size = block_desc->OpsSize();
    for (size_t op_idx = 0; op_idx < op_size; ++op_idx) {
      auto* op_desc = block_desc->GetOp<cpp::OpDesc>(op_idx);
      auto op_type = op_desc->Type();
      VLOG(4) << "create Op [" << op_type << "]";
      auto op = LiteOpRegistry::Global().Create(op_type);
      CHECK(op) << "no Op found for " << op_type;
      if (op_type == "while") {
        static_cast<operators::WhileOp*>(op.get())->SetProgramDesc(
            program_desc);
      } else if (op_type == "conditional_block") {
        static_cast<operators::ConditionalBlockOp*>(op.get())->SetProgramDesc(
            program_desc);
      } else if (op_type == "subgraph") {
        static_cast<operators::SubgraphOp*>(op.get())->SetProgramDesc(
            program_desc);
      }
      op->Attach(*op_desc, exec_scope_);
      ops_[block_idx].emplace_back(std::move(op));
    }
  }
}

void Program::PrepareWorkspace(
    const std::shared_ptr<cpp::ProgramDesc>& program_desc,
    const std::vector<std::string>& vars_to_clone) {
  CHECK(!exec_scope_) << "Duplicate PrepareWorkspace found";
  exec_scope_ = &scope_->NewScope();
  // Create Feed and Fetch var.
  scope_->Var("feed")->GetMutable<std::vector<lite::Tensor>>();
  scope_->Var("fetch")->GetMutable<std::vector<lite::Tensor>>();
  vars_.push_back("feed");
  vars_.push_back("fetch");

  auto VarDescType2PrecisionType =
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
        LOG(WARNING) << "Unable to convert var desc type("
                     << static_cast<int>(type) << ") to precision type!";
        return PRECISION(kUnk);
    }
  };

  auto block_size = program_desc->BlocksSize();
  CHECK(block_size);
  for (size_t block_idx = 0; block_idx < block_size; ++block_idx) {
    auto* block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);
    auto var_size = block_desc->VarsSize();
    for (size_t var_idx = 0; var_idx < var_size; ++var_idx) {
      auto* var_desc = block_desc->GetVar<cpp::VarDesc>(var_idx);
      const auto& var_name = var_desc->Name();
      const auto& var_type = var_desc->GetType();
      if (!var_desc->Persistable()) {
        vars_.push_back(var_name);
        auto* var = exec_scope_->Var(var_name);
        VLOG(4) << "Var " << var_name << " in block " << block_idx;
        VLOG(4) << " - type " << static_cast<int>(var_type);
        if (var_type == lite::VarDescAPI::Type::LOD_TENSOR) {
          const auto& var_data_type =
              VarDescType2PrecisionType(var_desc->GetDataType());
          if (var_data_type != PRECISION(kUnk)) {
            var_type_map_[var_name] = LiteType::GetTensorTy(
                TARGET(kUnk), var_data_type, DATALAYOUT(kUnk));
          }
          VLOG(4) << " - data type " << static_cast<int>(var_data_type);
          // Create the tensor with the shape from var desc, it's convenient to
          // the graph analysis in the passes, but you should resize the tensor
          // with the real shape before accessing its data, because the
          // var_shape may be [-1,3,224,224]
          const auto& var_shape = var_desc->GetShape();
          auto* tensor = var->GetMutable<lite::Tensor>();
          if (tensor->dims().empty() && !var_shape.empty()) {
            tensor->Resize(var_shape);
            VLOG(4) << " - dims " << tensor->dims().repr();
          }
        } else if (var_type == lite::VarDescAPI::Type::LOD_TENSOR_ARRAY) {
          var_type_map_[var_name] = LiteType::GetTensorListTy(
              TARGET(kUnk), PRECISION(kUnk), DATALAYOUT(kUnk));
        }
      } else {
        if (var_name == "feed" || var_name == "fetch") continue;
        weights_.push_back(var_name);
        scope_->Var(var_name);
      }
    }
  }

  for (auto var_name : vars_to_clone) {
    exec_scope_->LocalVar(var_name);
    auto* tensor = scope_->Var(var_name)->GetMutable<Tensor>();
    auto* sub_tensor = exec_scope_->Var(var_name)->GetMutable<Tensor>();
    sub_tensor->CopyDataFrom(*tensor);
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

#ifdef LITE_WITH_PROFILE
  if (first_epoch_for_profiler_) {
    kernel_->SetIsKernelTest(false);
    SetProfileRuntimeOpInfo(profiler_->GetOpCharacter(profile_id_));
    first_epoch_for_profiler_ = false;
  }
#endif
}

STL::ostream& operator<<(STL::ostream& os, const Instruction& other) {
  os << other.kernel_->summary() << "\t(" << other.kernel_->doc() << ")";
  return os;
}

}  // namespace lite
}  // namespace paddle
