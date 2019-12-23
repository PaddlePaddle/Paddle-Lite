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
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/model_parser/cpp/program_desc.h"

namespace paddle {
namespace lite {

static const char kKernelTypeAttr[] = "__@kernel_type_attr@__";

// A program is used to represent a code program, in Paddle, a code program
// contains:
// - main block, which is a list of OpLite
// - scope: which contains all the weights
struct Program {
 public:
  explicit Program(const std::shared_ptr<Scope>& root) { scope_ = root; }
  Program(const cpp::ProgramDesc& desc,
          const std::shared_ptr<Scope>& root,
          const std::vector<Place>& valid_places)
      : scope_(root), valid_places_(valid_places), desc_(desc) {
    CHECK(scope_) << "scope should be init first";
    VLOG(4) << "prepare work";
    PrepareWorkspace(desc);
    VLOG(4) << "build desc";
    Build(desc);
    VLOG(4) << "build desc finished";
  }

  std::unique_ptr<Program> Clone() const {
    std::unique_ptr<Program> res(new Program(desc_, scope_, valid_places_));
    return res;
  }

  const std::list<std::string>& weights() const { return weights_; }
  const std::list<std::string>& tmp_vars() const { return tmp_vars_; }
  std::list<std::string>* mutable_weights() { return &weights_; }
  std::list<std::string>* mutable_tmp_vars() { return &tmp_vars_; }

  const std::list<std::shared_ptr<OpLite>>& ops() const { return ops_; }
  std::list<std::shared_ptr<OpLite>>* mutable_ops() { return &ops_; }

  lite::Scope* exec_scope() { return exec_scope_; }
  lite::Scope* scope() { return scope_.get(); }

  const std::unordered_map<std::string, PrecisionType>& var_data_type() const {
    return var_data_type_;
  }

 private:
  // Build from a program and scope.
  void Build(const cpp::ProgramDesc& program);
  // Create temporary variables.
  void PrepareWorkspace(const cpp::ProgramDesc& program);

 private:
  std::unordered_map<std::string, PrecisionType> var_data_type_;
  std::list<std::string> tmp_vars_;
  std::list<std::string> weights_;
  std::list<std::shared_ptr<OpLite>> ops_;
  // the scope to run the kernels, NOTE this is the execution scope.
  std::shared_ptr<lite::Scope> scope_;
  std::vector<Place> valid_places_;
  // Runtime scope.
  lite::Scope* exec_scope_{};
  cpp::ProgramDesc desc_;
};

struct Instruction {
  Instruction(const std::shared_ptr<OpLite>& op,
              std::unique_ptr<KernelBase>&& kernel)
      : op_(op), kernel_(std::move(kernel)) {}

  // Run the instruction.
  void Run();

  friend STL::ostream& operator<<(STL::ostream& os, const Instruction& other);

  const OpLite* op() const { return op_.get(); }
  const KernelBase* kernel() const { return kernel_.get(); }
  KernelBase* mutable_kernel() { return kernel_.get(); }

#ifdef LITE_WITH_PROFILE
  void set_profiler(profile::Profiler* profiler) {
    profiler_ = profiler;
    if (op_->Type() != "feed" && op_->Type() != "fetch") {
      profile::OpCharacter ch;
      ch.target = kernel()->target();
      ch.op_type = op_->Type();
      ch.kernel_name = kernel()->name();
      profile_id_ = profiler->NewTimer(ch);
      kernel_->SetProfiler(profiler_, profile_id_);
    }
  }
#endif

 private:
  std::shared_ptr<OpLite> op_;
  std::unique_ptr<KernelBase> kernel_;
  bool first_epoch_{true};
  bool has_run_{false};

#ifdef LITE_WITH_PROFILE
  profile::Profiler* profiler_;
  int profile_id_{-1};
#endif  // LITE_WITH_PROFILE
};

/*
 * A program contains kernels for runtime.
 */
class LITE_API RuntimeProgram {
 public:
  explicit RuntimeProgram(std::vector<Instruction>&& insts)
      : instructions_(std::move(insts)) {
    if (instructions_.empty()) {
      LOG(FATAL) << "no instructions";
    }
#ifdef LITE_WITH_PROFILE
    set_profiler();
#endif
  }
  ~RuntimeProgram() {
#ifdef LITE_WITH_PROFILE
    LOG(INFO) << "\n" << profiler_.Summary();
#endif  // LITE_WITH_PROFILE
  }

  void Run();

  void set_exec_scope(lite::Scope* x) { exec_scope_ = x; }
  lite::Scope* exec_scope() { return exec_scope_; }

  size_t num_instructions() const { return instructions_.size(); }

  const std::vector<Instruction>& instructions() const { return instructions_; }

  // `SaveOpInfosToProgram` will update the op list(ops_) of the block 0
  // in ProgramDesc.
  void SaveOpInfosToProgram(cpp::ProgramDesc* desc);

  // `UpdateVarsOfProgram` will update the var list(vars_) of the block 0 in
  // ProgramDesc. Namely, if a new var created in some passes, its var_desc will
  // be added in vars_.
  void UpdateVarsOfProgram(cpp::ProgramDesc* desc);

 private:
  RuntimeProgram(const RuntimeProgram&) = delete;
  std::vector<Instruction> instructions_;
  lite::Scope* exec_scope_{};

#ifdef LITE_WITH_PROFILE
  profile::Profiler profiler_;
  void set_profiler() {
    for (auto i = instructions_.begin(); i != instructions_.end(); ++i) {
      i->set_profiler(&profiler_);
    }
  }
#endif
};

}  // namespace lite
}  // namespace paddle
