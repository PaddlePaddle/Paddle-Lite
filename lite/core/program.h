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
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/model_parser/cpp_desc.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#ifdef LITE_WITH_OPENCL
#include "lite/backends/opencl/cl_runtime.h"
#endif

namespace paddle {
namespace lite {

static const char kKernelTypeAttr[] = "__@kernel_type_attr@__";

// A program is used to represent a code program, in Paddle, a code program
// contains:
// - main block, which is a list of OpLite
// - scope: which contains all the weights
struct Program {
 public:
  explicit Program(const std::shared_ptr<Scope>& root_scope) {
    scope_ = root_scope;
  }
  Program(const std::shared_ptr<cpp::ProgramDesc>& program_desc,
          const std::shared_ptr<Scope>& root_scope,
          const std::vector<Place>& valid_places,
          const std::vector<std::string>& var_names = {})
      : scope_(root_scope), valid_places_(valid_places) {
    CHECK(scope_) << "scope should be init first";
    VLOG(4) << "prepare work";
    PrepareWorkspace(program_desc, var_names);
    VLOG(4) << "build desc";
    Build(program_desc);
    VLOG(4) << "build desc finished";
  }

  std::unique_ptr<Program> Clone() const {
    return std::unique_ptr<Program>(new Program(scope_));
  }

  const std::list<std::string>& weights() const { return weights_; }
  const std::list<std::string>& vars() const { return vars_; }
  std::list<std::string>* mutable_weights() { return &weights_; }
  std::list<std::string>* mutable_vars() { return &vars_; }

  const std::list<std::shared_ptr<OpLite>>& ops(
      int block_idx = kRootBlockIdx) const {
    return ops_[block_idx];
  }
  std::list<std::shared_ptr<OpLite>>* mutable_ops(
      int block_idx = kRootBlockIdx) {
    return &ops_[block_idx];
  }

  size_t block_size() { return ops_.size(); }

  Scope* exec_scope() { return exec_scope_; }
  Scope* scope() { return scope_.get(); }

  const std::map<std::string, const Type*>& var_type_map() const {
    return var_type_map_;
  }

  std::vector<std::string> getBlockOpsOrder(int block_idx) {
    std::vector<std::string> ret;
    for (auto& it : ops_[block_idx]) ret.push_back(it->op_info()->Type());
    return ret;
  }

 private:
  // Build from a program and scope.
  void Build(const std::shared_ptr<cpp::ProgramDesc>& program_desc);
  // Create temporary variables.
  void PrepareWorkspace(const std::shared_ptr<cpp::ProgramDesc>& program_desc,
                        const std::vector<std::string>& vars_to_clone = {});

 private:
  std::map<std::string, const Type*> var_type_map_;
  std::list<std::string> vars_;
  std::list<std::string> weights_;
  std::vector<std::list<std::shared_ptr<OpLite>>> ops_;
  // the scope to run the kernels, NOTE this is the execution scope.
  std::shared_ptr<Scope> scope_;
  std::vector<Place> valid_places_;
  // Runtime scope.
  Scope* exec_scope_{};
};

struct Instruction {
  Instruction(const std::shared_ptr<OpLite>& op,
              std::unique_ptr<KernelBase>&& kernel)
      : op_(op), kernel_(std::move(kernel)) {
    std::string op_type = op->Type();
    if (op_type == "feed" || op_type == "fetch") {
      is_feed_fetch_op_ = true;
    }
  }

  // Run the instruction.
  void Run();
#ifdef LITE_WITH_METAL
  void SaveOutput();
#endif

  friend STL::ostream& operator<<(STL::ostream& os, const Instruction& other);

  const OpLite* op() const { return op_.get(); }
  const KernelBase* kernel() const { return kernel_.get(); }
  KernelBase* mutable_kernel() { return kernel_.get(); }

  bool is_feed_fetch_op() const { return is_feed_fetch_op_; }

#ifdef LITE_WITH_OPENCL
  void Flush(const int inst_idx) const {
    if (TargetType::kOpenCL == kernel_->target()) {
      CLRuntime::Global()->Flush(inst_idx);
    }
  }
#endif

#ifdef LITE_WITH_PROFILE
  void set_profiler(profile::Profiler* profiler) {
    profiler_ = profiler;
#if !defined(LITE_WITH_METAL)
    if (op_->Type() != "feed" && op_->Type() != "fetch") {
#endif
      profile::OpCharacter ch;
      ch.op_lite = static_cast<void*>(const_cast<paddle::lite::OpLite*>(op()));
      ch.target = kernel()->target();
      ch.op_type = op_->Type();
      ch.kernel_name = kernel()->name();
      ch.kernel_attr = kernel()->name().substr(ch.op_type.size() + 1,
                                               kernel()->name().size());
      // append `ch.kernel_func_name` in StopTiming
      profile_id_ = profiler->NewTimer(ch);
      kernel_->SetProfiler(profiler_, profile_id_);
#if !defined(LITE_WITH_METAL)
    }
#endif
  }

  void SetProfileRuntimeOpInfo(paddle::lite::profile::OpCharacter* ch) {
    CHECK(ch != nullptr) << "OpCharacter should not be nullptr.";
    auto* op_lite = static_cast<paddle::lite::OpLite*>(ch->op_lite);
    CHECK(op_lite != nullptr) << "op_lite should not be nullptr.";
    op_lite->GetOpRuntimeInfo(ch);
  }
#endif

 private:
  std::shared_ptr<OpLite> op_;
  std::unique_ptr<KernelBase> kernel_;
  bool is_feed_fetch_op_{false};
  bool first_epoch_{true};
  bool has_run_{false};

#ifdef LITE_WITH_PROFILE
  profile::Profiler* profiler_;
  int profile_id_{-1};
  bool first_epoch_for_profiler_{true};
#endif  // LITE_WITH_PROFILE
};

/*
 * A program contains kernels for runtime.
 */
class LITE_API RuntimeProgram {
 public:
  explicit RuntimeProgram(std::vector<std::vector<Instruction>>&& insts)
      : instructions_(std::move(insts)) {
    Init();
  }
  explicit RuntimeProgram(
      const std::shared_ptr<const cpp::ProgramDesc>& program_desc,
      Scope* exec_scope,
      int block_idx = kRootBlockIdx,
      bool use_precision_low = false);
  bool use_precision_low_ = false;
  ~RuntimeProgram() {
#ifdef LITE_WITH_OPENCL
    // when has opencl kernel try to save params
    if (has_opencl_kernel_) {
      // save program kernel cache & tuned params
      CLRuntime::Global()->SaveProgram();
      CLRuntime::Global()->SaveTuned();
    }
#endif  // LITE_WITH_OPENCL
#ifdef LITE_WITH_PROFILE
    // exclude data of first epoch
    LOG(INFO) << "\n" << profiler_.Summary(profile::Type::kDispatch, false, 1);
    // exclude data of 10 warm-up
    LOG(INFO) << "\n" << profiler_.Summary(profile::Type::kCreate);
    LOG(INFO) << "\n" << profiler_.Summary(profile::Type::kDispatch);
#endif  // LITE_WITH_PROFILE
  }

  void Init() {
    if (instructions_.empty()) {
      LOG(FATAL) << "No instructions";
    }
#ifdef LITE_WITH_PROFILE
    set_profiler();
#endif
    for (auto& inst : instructions_[kRootBlockIdx]) {
      KernelBase* kernel = inst.mutable_kernel();
      if (kernel->target() == TARGET(kOpenCL)) {
#if defined(LITE_WITH_OPENCL)
        // mark has opencl kernel
        has_opencl_kernel_ = true;

        // init opencl runtime when first find opencl kernel.
        // when unique_opencl_ctx_ not init. init it
        if (!unique_opencl_ctx_) {
          // auto enable when opencl kernel is found.
          ClGlobalDelegate::Global().SetUseOpenCL(true);
          opencl_valid_ = paddle::lite::CLWrapper::Global()->OpenclLibFound() &&
                          paddle::lite::CLWrapper::Global()->DlsymSuccess() &&
                          CLRuntime::Global()->OpenCLAvaliableForDevice();
          // check opencl env valid.
          if (opencl_valid_) {
            // init opencl context
            std::unique_ptr<KernelContext> unique_opencl_ctx(
                new KernelContext());
            unique_opencl_ctx_ = std::move(unique_opencl_ctx);
            (*unique_opencl_ctx_).As<OpenCLContext>().InitOnce();
          } else {
            LOG(FATAL) << "Check opencl env failed. opencl_valid:"
                       << opencl_valid_;
          }
        }

        // check valid and copy shared context to kernel.
        if (opencl_valid_) {
          std::unique_ptr<KernelContext> ctx(new KernelContext());
          (*unique_opencl_ctx_)
              .As<OpenCLContext>()
              .CopySharedTo(&ctx->As<OpenCLContext>());
          kernel->SetContext(std::move(ctx));
        } else {
          // if gpu not support , fatal when user init gpu model.
          LOG(FATAL) << "OpenCl Valid: " << opencl_valid_;
        }
#endif
      } else if (kernel->target() == TARGET(kMetal)) {
#if defined(LITE_WITH_METAL)
        if (!metal_ctx_) {
          metal_ctx_ = std::make_unique<KernelContext>();
          (*metal_ctx_).As<MTLContext>().InitOnce();
        }
        std::unique_ptr<KernelContext> ctx(new KernelContext());
        (*metal_ctx_).As<MTLContext>().CopySharedTo(&ctx->As<MTLContext>());
        kernel->SetContext(std::move(ctx));
#endif
      } else {
        if (kernel != nullptr) {
          kernel->SetContext(
              ContextScheduler::Global().NewContext(kernel->target()));
        }
      }
    }
  }

  void Run();
#ifdef LITE_WITH_METAL
  void SaveOutput();
#endif

  void set_exec_scope(Scope* x) { exec_scope_ = x; }
  Scope* exec_scope() { return exec_scope_; }

  const std::vector<Instruction>& instructions(
      int block_idx = kRootBlockIdx) const {
    return instructions_[block_idx];
  }

  std::vector<Instruction>* mutable_instructions(
      int block_idx = kRootBlockIdx) {
    return &instructions_[block_idx];
  }

  size_t block_size() { return instructions_.size(); }

  void set_version(const int64_t version) { version_ = version; }

  const int64_t get_version() const { return version_; }

#ifndef LITE_ON_TINY_PUBLISH
  // Update the ops and vars of all of blocks to the given program_desc
  // according to the instructions
  void SaveRuntimProgramIntoProgramDesc(
      std::shared_ptr<cpp::ProgramDesc> program_desc);
#endif

#ifdef LITE_WITH_METAL
  void ConfigMetalContext(std::string lib_path,
                          bool use_mps = false,
                          bool use_aggressive = false,
                          bool use_memory_reuse_ = false,
                          void* device = nullptr);
#endif

 private:
  RuntimeProgram(const RuntimeProgram&) = delete;
  std::vector<std::vector<Instruction>> instructions_;
  Scope* exec_scope_{};
  int64_t version_{0};

#ifdef LITE_WITH_OPENCL
  bool opencl_valid_{false};
  bool has_opencl_kernel_{false};
  std::unique_ptr<KernelContext> unique_opencl_ctx_;
#endif
#ifdef LITE_WITH_METAL
  std::unique_ptr<KernelContext> metal_ctx_{nullptr};
#endif

#ifdef LITE_WITH_PROFILE
  profile::Profiler profiler_;
  void set_profiler() {
    for (auto& inst : instructions_[kRootBlockIdx]) {
      inst.set_profiler(&profiler_);
    }
  }
#endif
};

}  // namespace lite
}  // namespace paddle
