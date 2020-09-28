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
#ifdef LITE_WITH_NVTX
#include "lite/backends/cuda/nvtx_wrapper.h"
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

  friend STL::ostream& operator<<(STL::ostream& os, const Instruction& other);

  const OpLite* op() const { return op_.get(); }
  const KernelBase* kernel() const { return kernel_.get(); }
  KernelBase* mutable_kernel() { return kernel_.get(); }

  bool is_feed_fetch_op() const { return is_feed_fetch_op_; }

#ifdef LITE_WITH_CUDA
  bool need_sync() const {
    if (kernel_->target() == TargetType::kCUDA) {
      return kernel_->mutable_context()->As<CUDAContext>().need_sync();
    } else {
      // the io_copy kernel has synced, so cpu kernels don't need sync..
      return false;
    }
  }
  void Sync() const { kernel_->mutable_context()->As<CUDAContext>().Sync(); }
#endif

#ifdef LITE_WITH_PROFILE
  void set_profiler(profile::Profiler* profiler) {
    profiler_ = profiler;
#ifndef LITE_WITH_FPGA
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
#ifndef LITE_WITH_FPGA
    }
#endif
  }

  void SetProfileRuntimeOpInfo(paddle::lite::profile::OpCharacter* ch) {
    auto* op_lite = static_cast<paddle::lite::OpLite*>(ch->op_lite);
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
      int block_idx = kRootBlockIdx);
  ~RuntimeProgram() {
#ifdef LITE_WITH_PROFILE
    LOG(INFO) << "\n" << profiler_.Summary(profile::Type::kCreate);
    LOG(INFO) << "\n" << profiler_.Summary(profile::Type::kDispatch);
#endif  // LITE_WITH_PROFILE
  }

  void Init() {
    if (instructions_.empty()) {
      LOG(FATAL) << "no instructions";
    }
#ifdef LITE_WITH_PROFILE
    set_profiler();
#endif
#ifdef LITE_WITH_NVTX
    const NVTXAnnotator& annotator = NVTXAnnotator::Global();
    for (auto& inst : instructions_[kRootBlockIdx]) {
      NVTXRangeAnnotation annotation = annotator.AnnotateBlock();
      register_layer_names_.push_back(annotator.RegisterString(
          const_cast<paddle::lite::OpLite*>(inst.op())->Type().c_str()));
    }
    register_layer_names_.push_back(annotator.RegisterString("one_loop"));
#endif
  }

  void Run();

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

  // Update the ops and vars of all of blocks to the given program_desc
  // according to the instructions
  void SaveToProgram(std::shared_ptr<cpp::ProgramDesc> program_desc);

 private:
  RuntimeProgram(const RuntimeProgram&) = delete;
  std::vector<std::vector<Instruction>> instructions_;
  Scope* exec_scope_{};

#ifdef LITE_WITH_PROFILE
  profile::Profiler profiler_;
  void set_profiler() {
    for (auto& inst : instructions_[kRootBlockIdx]) {
      inst.set_profiler(&profiler_);
    }
  }
#endif
#ifdef LITE_WITH_NVTX
  std::vector<nvtxStringHandle_t> register_layer_names_;
#endif
};

}  // namespace lite
}  // namespace paddle
