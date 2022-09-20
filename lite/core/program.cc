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

#ifdef ENABLE_ARM_FP16
#include "lite/backends/arm/math/fp16/funcs_fp16.h"
#endif
#include "lite/model_parser/cpp_desc.h"
#include "lite/operators/conditional_block_op.h"
#include "lite/operators/subgraph_op.h"
#include "lite/operators/while_op.h"
#ifdef LITE_WITH_PRECISION_PROFILE
#include "lite/core/profile/precision_profiler.h"
#endif
#ifdef LITE_WITH_FPGA
#include "lite/backends/fpga/monitor.hpp"
#endif

namespace paddle {
namespace lite {
#ifndef LITE_ON_TINY_PUBLISH
namespace {
// Verify the validity of ProgramDesc
void CheckProgramDescValidity(std::shared_ptr<cpp::ProgramDesc> program_desc,
                              int inst_block_size) {
  CHECK(program_desc) << "Error, program_desc is nullptr";
  auto block_size = program_desc->BlocksSize();
  CHECK_GT(block_size, 0) << "No block exists in current program_desc";
  // TODD(hong19860320) Only support updating the block desc which already
  // exists in the origin program desc
  CHECK_LE(block_size, inst_block_size) << "Invalid block size, expected (0,"
                                        << inst_block_size << "] but got "
                                        << block_size;
}

std::map<std::string, cpp::VarDesc> ClearBlockDescInfo(
    cpp::BlockDesc* block_desc) {
  std::map<std::string, cpp::VarDesc> origin_var_maps;
  auto var_size = block_desc->VarsSize();
  for (size_t var_idx = 0; var_idx < var_size; ++var_idx) {
    auto v = block_desc->GetVar<cpp::VarDesc>(var_idx);
    origin_var_maps.emplace(v->Name(), *v);
  }
  // Update the ops and vars for each block according to the instructions
  block_desc->ClearVars();
  block_desc->ClearOps();
  return origin_var_maps;
}

void UpdatePersistableVarDesc(cpp::VarDesc* var,
                              const cpp::VarDesc& previous_var_desc,
                              const std::string& var_name,
                              Scope* scope) {
  var->SetType(previous_var_desc.GetType());
  var->SetPersistable(previous_var_desc.Persistable());
  if (previous_var_desc.GetType() == cpp::VarDesc::Type::LOD_TENSOR) {
    if (var != nullptr) {
      auto tensor = scope->FindVar(var_name)->GetMutable<Tensor>();
      if (tensor != nullptr && tensor->persistable()) {
        var->SetPersistable(tensor->persistable());
      }
    }
  }
  if (var_name != "feed" && var_name != "fetch") {
    var->SetShape(previous_var_desc.GetShape());
    var->SetDataType(previous_var_desc.GetDataType());
  }
}

void UpdateVarDescFromTensorInfo(cpp::VarDesc* var,
                                 const std::string& var_name,
                                 const std::string& op_type,
                                 Scope* scope) {
  var->SetType(cpp::VarDesc::Type::LOD_TENSOR);
  auto tensor = scope->FindVar(var_name)->GetMutable<Tensor>();
  var->SetPersistable(tensor->persistable());
  // Move the persistable var from exec scope to the root scope
  auto root_scope = scope->MutableParent();
  if (tensor->persistable() && root_scope != scope &&
      !root_scope->FindLocalVar(var_name)) {
    // Find or create new var in root scope
    auto root_tensor = root_scope->LocalVar(var_name)->GetMutable<Tensor>();
    if (root_tensor != tensor) {
      root_tensor->CopyDataFrom(*tensor);
      scope->DeleteLocalVar(var_name);
    }
  }

  if (var_name != "feed" && var_name != "fetch") {
    var->SetShape(tensor->dims().data());
    auto precision = tensor->precision();
    switch (precision) {
#define SET_DATATYPE(precision__, data_type) \
  case PrecisionType::precision__:           \
    var->SetDataType(data_type);             \
    break
      SET_DATATYPE(kBool, VarDescAPI::VarDataType::BOOL);
      SET_DATATYPE(kFP16, VarDescAPI::VarDataType::FP16);
      SET_DATATYPE(kFloat, VarDescAPI::VarDataType::FP32);
      SET_DATATYPE(kFP64, VarDescAPI::VarDataType::FP64);
      SET_DATATYPE(kUInt8, VarDescAPI::VarDataType::UINT8);
      SET_DATATYPE(kInt8, VarDescAPI::VarDataType::INT8);
      SET_DATATYPE(kInt16, VarDescAPI::VarDataType::INT16);
      SET_DATATYPE(kInt32, VarDescAPI::VarDataType::INT32);
      SET_DATATYPE(kInt64, VarDescAPI::VarDataType::INT64);
      SET_DATATYPE(kUnk, VarDescAPI::VarDataType::FP32);
      SET_DATATYPE(kAny, VarDescAPI::VarDataType::FP32);
#undef SET_DATATYPE
      default:
        LOG(FATAL) << "Unknown precision type " << PrecisionToStr(precision)
                   << " for var " << var_name << " in op " << op_type;
    }
  }
}

void UpdateVarDescFromTensorListInfo(cpp::VarDesc* var,
                                     const std::string& var_name,
                                     const std::string& op_type,
                                     Scope* scope) {
  var->SetType(cpp::VarDesc::Type::LOD_TENSOR_ARRAY);
  var->SetPersistable(false);
}

void UpdateVarDescFromStepScopeInfo(cpp::VarDesc* var,
                                    const std::string& var_name,
                                    const std::string& op_type,
                                    Scope* scope) {
  var->SetType(cpp::VarDesc::Type::STEP_SCOPES);
  var->SetPersistable(false);
}

const Type* GetVariableDeclTypeFromOpInfo(const std::string& var_name,
                                          const OpInfo* op_info,
                                          KernelBase* kernel) {
  std::string arg_name;
  const Type* decl_type;
  if (op_info->GetInputArgname(var_name, &arg_name)) {
    decl_type = kernel->GetInputDeclType(arg_name);
  } else {
    op_info->GetOutputArgname(var_name, &arg_name);
    decl_type = kernel->GetOutputDeclType(arg_name);
  }
  return decl_type;
}

void AddOpDescFromOpInfo(std::shared_ptr<cpp::ProgramDesc> program_desc,
                         size_t block_idx,
                         Instruction* inst) {
  auto* block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);
  auto op_desc = block_desc->AddOp<cpp::OpDesc>();
  auto* op = const_cast<OpLite*>(inst->op());
  auto* kernel = inst->mutable_kernel();

  auto* op_info = op->op_info();
  *op_desc = *op_info;
  op_desc->SetAttr(kKernelTypeAttr, kernel->SerializedKernelType());
  auto* scope = op->scope();
  auto op_type = op_info->Type();
  // Update subgraph op
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

void AddVariableDescFromOpInfo(
    std::shared_ptr<cpp::ProgramDesc> program_desc,
    size_t block_idx,
    Instruction* inst,
    std::set<std::string>* already_added_vars,
    const std::map<std::string, cpp::VarDesc>& origin_var_maps) {
  auto* block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);

  auto* op = const_cast<OpLite*>(inst->op());
  auto* kernel = inst->mutable_kernel();
  auto* op_info = op->op_info();
  auto* scope = op->scope();
  auto op_type = op_info->Type();

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
    if (already_added_vars->count(var_name)) continue;
    auto* v = block_desc->AddVar<cpp::VarDesc>();
    v->SetName(var_name);

    auto* var = scope->FindVar(var_name);
    auto it = origin_var_maps.find(var_name);

    if (it != origin_var_maps.end() && (it->second.Persistable())) {
      UpdatePersistableVarDesc(v, it->second, var_name, scope);
    } else {
      auto* decl_type =
          GetVariableDeclTypeFromOpInfo(var_name, op_info, kernel);
      if (decl_type->IsTensor() && var->IsType<lite::Tensor>()) {
        UpdateVarDescFromTensorInfo(v, var_name, op_type, scope);
      } else if (decl_type->IsTensorList() ||
                 var->IsType<std::vector<lite::Tensor>>()) {
        UpdateVarDescFromTensorListInfo(v, var_name, op_type, scope);
      } else if (decl_type->IsStepScope() &&
                 var->IsType<std::vector<lite::Scope*>>()) {
        UpdateVarDescFromStepScopeInfo(v, var_name, op_type, scope);
      } else {
        LOG(FATAL) << "Unsupported decl type " << *decl_type << " for var "
                   << var_name << " in op " << op_type;
      }
    }
    already_added_vars->insert(var_name);
  }
}

}  // namespace

void RuntimeProgram::SaveRuntimProgramIntoProgramDesc(
    std::shared_ptr<cpp::ProgramDesc> program_desc) {
  CheckProgramDescValidity(program_desc, instructions_.size());
  size_t block_size = program_desc->BlocksSize();
  program_desc->SetVersion(get_version());
  for (size_t block_idx = 0; block_idx < block_size; ++block_idx) {
    std::set<std::string> already_added_vars;
    const std::map<std::string, cpp::VarDesc> origin_var_maps =
        ClearBlockDescInfo(program_desc->GetBlock<cpp::BlockDesc>(block_idx));
    for (auto& inst : instructions_[block_idx]) {
      AddVariableDescFromOpInfo(
          program_desc, block_idx, &inst, &already_added_vars, origin_var_maps);
      // Replace all of origin ops with the instructions
      AddOpDescFromOpInfo(program_desc, block_idx, &inst);
    }
  }
}
#endif

// Create runtime program from sub_block desc according to block_idx and
// program_desc, which is used for while/conditional_block/subgraph op.
RuntimeProgram::RuntimeProgram(
    const std::shared_ptr<const cpp::ProgramDesc>& program_desc,
    Scope* exec_scope,
    int block_idx,
    bool use_precision_low)
    : exec_scope_(exec_scope) {
  CHECK(program_desc);
  auto block_size = program_desc->BlocksSize();
  CHECK(block_size) << "No block found!";
  CHECK(block_idx >= 0 && block_idx < block_size)
      << "Invalid block index, expected [0," << (block_size - 1) << "] but got "
      << block_idx;
  auto block_desc = program_desc->GetBlock<cpp::BlockDesc>(block_idx);
  instructions_.resize(kRootBlockIdx + 1);
  auto op_size = block_desc->OpsSize();

  int low_precision = 1;
  std::string old_op;
  use_precision_low_ = use_precision_low;
  low_precision = use_precision_low_ ? 1 : 0;

  for (size_t op_idx = 0; op_idx < op_size; op_idx++) {
    auto op_desc = block_desc->GetOp<cpp::OpDesc>(op_idx);
    CHECK(op_desc);
    std::string op_type = op_desc->Type();
    // if (op_type == "feed" || op_type == "fetch") continue;
    // Create op and pick up the best kernel
    auto op = LiteOpRegistry::Global().Create(op_type);

// Error message: if current kernel is not supported, WITH_EXTRA lib is
// suggested.
#ifndef LITE_BUILD_EXTRA
    std::string ops_error_message =
        "\nError: Please use Paddle-Lite lib with all ops, which is marked "
        "with "
        "`with_extra`. Current lib is of tiny_publish, in which only basic "
        "ops are included and we can not create operator '" +
        op_type +
        "'.\n Two ways are suggested to get Paddle-Lite lib with all ops:\n    "
        "1. Download pre-compiled lib which is marked with `with_extra`.\n    "
        "2. Compile Paddle-Lite with command `--with_extra=ON`.";
#else
    std::string ops_error_message =
        "\nError: This model is not supported, because operator '" + op_type +
        "' is not supported by Paddle-Lite.";
#endif
    CHECK(op) << ops_error_message;

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

// Error message: if current kernel is not supported, WITH_EXTRA lib is
// suggested.
#ifndef LITE_BUILD_EXTRA
      std::string kernels_error_message =
          "\nError: Please use Paddle-Lite lib with all ops, which is marked "
          "with "
          "`with_extra`. Current lib is of tiny_publish, in which only basic "
          "kernels "
          "are included and we can not create kernel for '" +
          op_type +
          "'.\n Two ways are suggested to get Paddle-Lite lib with all "
          "kernels:\n    "
          "1. Download pre-commit lib which is marked with `with_extra`.\n    "
          "2. "
          "Compile Paddle-Lite with command `--with_extra=ON`.";
#else
      std::string kernels_error_message =
          "\nError: This model is not supported, because kernel for '" +
          op_type + "' is not supported by Paddle-Lite.";
#endif

#ifdef ENABLE_ARM_FP16
      if (lite::DeviceInfo::Global().has_fp16() && low_precision == 1) {
        if (op_type != "feed" && op_type != "fetch") {
          if (place.precision == PRECISION(kFloat)) {
            place.precision = PRECISION(kFP16);
          } else if (place.precision == PRECISION(kAny)) {
            place.precision = PRECISION(kFP16);
          }
        }
        // transfer weight to fp16
        std::vector<std::string> fp16_ops{"conv2d",
                                          "depthwise_conv2d",
                                          "conv2d_transpose",
                                          "fc",
                                          "mul",
                                          "matmul",
                                          "matmul_v2",
                                          "gru",
                                          "sequence_conv",
                                          "elementwise_add",
                                          "elementwise_sub",
                                          "elementwise_div",
                                          "elementwise_mul",
                                          "prelu"};
        typedef __fp16 float16_t;
        auto iter = std::find(fp16_ops.begin(), fp16_ops.end(), op_type);
        if (iter != fp16_ops.end()) {
          auto input_names = op_desc->input_vars();

          for (auto& input_name : input_names) {
            if (input_name == "") continue;
            if (input_name == "feed" || input_name == "fetch") continue;
            if (exec_scope_->FindVar(input_name)) {
              if (!exec_scope_->FindVar(input_name)->IsType<lite::Tensor>())
                continue;

              auto input_tensor =
                  exec_scope_->Var(input_name)->GetMutable<lite::Tensor>();

              if (input_tensor->persistable()) {
                if (input_tensor->precision() != PRECISION(kFloat)) continue;

                input_tensor->set_precision(PRECISION(kFP16));
                Tensor tmp_tensor;
                tmp_tensor.CopyDataFrom(*input_tensor);
                input_tensor->clear();
                input_tensor->set_precision(PRECISION(kFP16));

                float16_t* fp_data = input_tensor->mutable_data<float16_t>();
                const float* in_data = tmp_tensor.data<float>();
                lite::arm::math::fp16::fp32_to_fp16(
                    in_data, fp_data, input_tensor->numel());
              }
            }
          }
        }
        auto kernels = op->CreateKernels({place});
        if (kernels.size() == 0 && place.target == TargetType::kARM) {
          place.target = TargetType::kHost;
          kernels = op->CreateKernels({place});
        }

        CHECK_GT(kernels.size(), 0) << kernels_error_message;
        auto it = std::find_if(kernels.begin(),
                               kernels.end(),
                               [&](std::unique_ptr<KernelBase>& it) {
                                 return it->alias() == alias;
                               });

        CHECK(it != kernels.end());
        kernel = std::move(*it);

        // insert calib op
        if (old_op == "feed" || op_type == "fetch") {
          auto input_names = op_desc->input_vars();
          std::vector<std::string> input_names_new;
          for (int i = 0; i < input_names.size(); i++) {
            std::string input_name = input_names[i];

            if (exec_scope_->Var(input_name)) {
              if (!exec_scope_->FindVar(input_name)->IsType<lite::Tensor>())
                continue;
              auto input_tensor =
                  exec_scope_->Var(input_name)->GetMutable<lite::Tensor>();

              if ((!input_tensor->persistable())) {
                cpp::OpDescWrite calib_desc;
                if (op_type == "fetch") {
                  calib_desc.SetType("calib_inplace");
                } else {
                  calib_desc.SetType("calib");
                }
                calib_desc.SetInput("Input", {input_name});
                auto output_name = input_name + "_calib";
                calib_desc.SetOutput("Out", {output_name});

                auto op_calib = LiteOpRegistry::Global().Create("calib");
                if (op_type == "fetch") {
                  op_calib = LiteOpRegistry::Global().Create("calib_inplace");
                }
                auto x_var = exec_scope_->FindVar(input_name);
                auto output_var = exec_scope_->Var(output_name);
                auto output_tensor =
                    exec_scope_->Var(output_name)->GetMutable<lite::Tensor>();

                op_calib->Attach(calib_desc, exec_scope_);
                Place fetch_calib_place;
                fetch_calib_place.target = TARGET(kARM);
                fetch_calib_place.layout = DATALAYOUT(kNCHW);
                fetch_calib_place.precision = PRECISION(kFP16);
                auto calib_kernels = op_calib->CreateKernels({place});
                if (op_type == "fetch") {
                  calib_kernels = op_calib->CreateKernels({fetch_calib_place});
                }
                std::string alias_calib;
                if (op_type != "fetch")
                  alias_calib = "fp32_to_fp16";
                else
                  alias_calib = "fp16_to_fp32";
                auto it = std::find_if(calib_kernels.begin(),
                                       calib_kernels.end(),
                                       [&](std::unique_ptr<KernelBase>& it) {
                                         return it->alias() == alias_calib;
                                       });
                std::unique_ptr<KernelBase> calib_kernel;
                if (it != calib_kernels.end()) {
                  calib_kernel = std::move(*it);

                  instructions_[kRootBlockIdx].emplace_back(
                      std::move(op_calib), std::move(calib_kernel));
                } else {
                }

                input_names_new.push_back(output_name);
              }
            }
          }
          if (op_type != "fetch") {
            cpp::OpDescWrite op_desc_write;
            op_desc_write.SetInput("Input", input_names_new);
            op->AttachInput(op_desc_write, exec_scope_);
            kernels = op->CreateKernels({place});
            if (kernels.size() == 0) {
              place.precision = PRECISION(kFloat);
              kernels = op->CreateKernels({place});
            }

            if (kernels.size() == 0 && place.target == TargetType::kARM) {
              place.target = TargetType::kHost;
              kernels = op->CreateKernels({place});
            }

            CHECK_GT(kernels.size(), 0) << kernels_error_message;
            auto it = std::find_if(kernels.begin(),
                                   kernels.end(),
                                   [&](std::unique_ptr<KernelBase>& it) {
                                     return it->alias() == alias;
                                   });

            CHECK(it != kernels.end());
            kernel = std::move(*it);
          }
        }

        old_op = op_type;
      } else {
#endif
        auto kernels = op->CreateKernels({place});
        if (kernels.size() == 0 && place.target == TargetType::kARM) {
          place.target = TargetType::kHost;
          kernels = op->CreateKernels({place});
        }
        CHECK_GT(kernels.size(), 0) << kernels_error_message;
        auto it = std::find_if(kernels.begin(),
                               kernels.end(),
                               [&](std::unique_ptr<KernelBase>& it) {
                                 return it->alias() == alias;
                               });
        CHECK(it != kernels.end());
        kernel = std::move(*it);
#ifdef ENABLE_ARM_FP16
      }
#endif

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
    instructions_[kRootBlockIdx].emplace_back(std::move(op), std::move(kernel));
  }
  Init();
}

#ifdef LITE_WITH_METAL
void RuntimeProgram::ConfigMetalContext(std::string lib_path,
                                        bool use_mps,
                                        bool use_aggressive,
                                        bool use_memory_reuse_,
                                        void* device) {
  if (!metal_ctx_) return;
  MetalContext* context = (*metal_ctx_).As<MTLContext>().context();
  context->set_metal_path(lib_path);
  context->set_use_mps(use_mps);
  context->set_use_aggressive(use_aggressive);
  context->set_metal_device(device);
  context->set_use_memory_reuse(use_memory_reuse_);
}

void RuntimeProgram::SaveOutput() {
  auto& insts = instructions_[kRootBlockIdx];
  for (auto& inst : insts) {
    inst.SaveOutput();
  }
}
#endif

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

#ifdef LITE_WITH_FPGA
  Monitor& monitor = Monitor::get_instance();
  monitor.inferStart();
#endif

  int idx = -1;

  auto& insts = instructions_[kRootBlockIdx];
  for (auto& inst : insts) {
    ++idx;
#if !defined(LITE_WITH_FPGA) && !defined(LITE_WITH_METAL)
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

#ifdef LITE_WITH_FPGA
    monitor.preRun(inst);
#endif

#ifdef LITE_WITH_OPENCL
    // delegate flush judgement to specify target , it is too heavy for Inst
    inst.Flush(idx);
#endif

    inst.Run();

#ifdef LITE_WITH_FPGA
    monitor.postRun(inst);
#endif

#ifdef LITE_WITH_PRECISION_PROFILE
#ifndef LITE_WITH_FPGA
    if (inst.op()->Type() != "while") {
      precision_profiler_summary +=
          inst_precision_profiler.GetInstPrecision(&inst);
    }
#endif
#endif  // LITE_WITH_PRECISION_PROFILE
  }

#ifdef LITE_WITH_METAL
  if (metal_ctx_) {
    MetalContext* wait_ctx = (*metal_ctx_).As<MTLContext>().context();
    wait_ctx->wait_all_completed();
  }
#endif

#ifdef LITE_WITH_PROFILE
  LOG(INFO) << "\n" << profiler_.Summary(profile::Type::kDispatch, false, 1);
#endif
#ifdef LITE_WITH_PRECISION_PROFILE
  LOG(INFO) << "\n"
            << precision_profiler_summary
            << inst_precision_profiler.GetSummaryTail();
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
      case lite::VarDescAPI::Type::BOOL:
        return PRECISION(kBool);
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
      case lite::VarDescAPI::Type::UINT8:
        return PRECISION(kUInt8);
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
      VLOG(4) << "Var " << var_name << " in block " << block_idx;
      VLOG(4) << " - type " << static_cast<int>(var_type);

#if defined(LITE_WITH_XPU) || defined(LITE_WITH_CUDA)
      if (!var_desc->Persistable()) {
#endif
        // Collect precision info into var_type_map_
        if (var_type == lite::VarDescAPI::Type::LOD_TENSOR) {
          const auto& var_data_type =
              VarDescType2PrecisionType(var_desc->GetDataType());
          if (var_data_type != PRECISION(kUnk)) {
            var_type_map_[var_name] = LiteType::GetTensorTy(
                TARGET(kUnk), var_data_type, DATALAYOUT(kUnk));
          }
          VLOG(4) << " - data type " << static_cast<int>(var_data_type);
        } else if (var_type == lite::VarDescAPI::Type::LOD_TENSOR_ARRAY) {
          var_type_map_[var_name] = LiteType::GetTensorListTy(
              TARGET(kUnk), PRECISION(kUnk), DATALAYOUT(kUnk));
        }
#if defined(LITE_WITH_XPU) || defined(LITE_WITH_CUDA)
      }
#endif
      // Create tensors or weights from variable description.
      if (!var_desc->Persistable()) {
        vars_.push_back(var_name);
        auto* var = exec_scope_->Var(var_name);
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
          tensor->set_precision(var_data_type);
          tensor->set_persistable(var_desc->Persistable());
        } else if (var_type == lite::VarDescAPI::Type::LOD_TENSOR_ARRAY) {
          var_type_map_[var_name] = LiteType::GetTensorListTy(
              TARGET(kUnk), PRECISION(kUnk), DATALAYOUT(kUnk));
          auto* tensor_array = var->GetMutable<std::vector<lite::Tensor>>();
          tensor_array->resize(0);
        } else if (var_type == lite::VarDescAPI::Type::STEP_SCOPES) {
          var->GetMutable<std::vector<lite::Scope*>>();
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

#ifdef LITE_WITH_METAL
void Instruction::SaveOutput() {
  if (kernel_) kernel_->SaveOutput();
}
#endif

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
    auto* op_ch = profiler_->GetOpCharacter(profile_id_);
    SetProfileRuntimeOpInfo(op_ch);
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
