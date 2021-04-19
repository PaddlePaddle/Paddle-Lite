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
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/mir/control_flow_op_shared_inputs_and_outputs_place_sync_pass.h"
#include "lite/core/mir/elimination/control_flow_op_unused_inputs_and_outputs_eliminate_pass.h"
#include "lite/core/mir/fp16_attribute_pass.h"
#include "lite/core/mir/generate_program_pass.h"
#include "lite/core/mir/pass_manager.h"
#include "lite/core/mir/pass_utils.h"
#include "lite/core/mir/post_quant_dynamic_pass.h"
#include "lite/core/mir/ssa_graph.h"
#include "lite/core/mir/static_kernel_pick_pass.h"
#include "lite/core/mir/type_target_cast_pass.h"
#include "lite/core/program.h"
#include "lite/core/types.h"
#include "lite/model_parser/model_parser.h"

namespace paddle {
namespace lite {

// TODO(hong1986032) Support the following passes for the subblocks
const std::set<std::string> kSubblockUnsupportedPasses(
    {"memory_optimize_pass"});

/*
 * lite::Optimizer optimize a program. It utilize the mir passes to analysis the
 * program and export an optimized program.
 */
class Optimizer {
 public:
  Optimizer(const std::vector<Place>& valid_places,
            core::KernelPickFactor kernel_pick_factor)
      : valid_places_(valid_places), kernel_pick_factor_(kernel_pick_factor) {
    CHECK(!valid_places.empty()) << "At least one valid_place should be set";
  }

  //! Append a pass to the optimizer.
  void AddPass(const std::string& pass_name) {
    mir::Pass* pass = mir::PassManager::Global().LookUp(pass_name);
    passes_.push_back(pass);
  }

  std::unique_ptr<RuntimeProgram> Run(Program&& program) {
    auto block_size = program.block_size();
    for (size_t block_idx = 0; block_idx < block_size; ++block_idx) {
      std::unique_ptr<mir::SSAGraph> graph;
      graph.reset(new mir::SSAGraph);
      graph->Build(program, valid_places_, block_idx);
      graph->SetValidPlaces(valid_places_);
      graphs_.emplace_back(std::move(graph));
    }

    SpecifyKernelPickTactic(kernel_pick_factor_);
    InitTargetTypeTransformPass();
    InitControlFlowOpUnusedInputsAndOutputsEliminatePass();
    InitControlFlowOpSharedInputsAndOutputsPlaceSyncPass();

    ApplyPasses(&graphs_);

    exec_scope_ = program.exec_scope();

    return GenRuntimeProgram(&graphs_);
  }

  const Scope* exec_scope() const { return exec_scope_; }

  std::unique_ptr<RuntimeProgram> GenRuntimeProgram(
      std::vector<std::unique_ptr<mir::SSAGraph>>* graphs) {
    auto pass = mir::PassManager::Global().LookUp<mir::GenerateProgramPass>(
        "generate_program_pass");
    for (auto& graph : *graphs) {
      pass->Apply(graph);
    }
    auto program = pass->GenProgram();
    CHECK(exec_scope_);
    program->set_exec_scope(exec_scope_);
    return program;
  }

  void InitTargetTypeTransformPass();

  void InitControlFlowOpUnusedInputsAndOutputsEliminatePass();

  void InitControlFlowOpSharedInputsAndOutputsPlaceSyncPass();

  Scope* exec_scope() { return exec_scope_; }

 protected:
  void SpecifyKernelPickTactic(core::KernelPickFactor factor);

  // Specify the passes and run them. NOTE legancy, to discarded latter.
  void RunPasses(const std::vector<std::string>& passes);

  // Run all the added passes.
  void ApplyPasses(std::vector<std::unique_ptr<mir::SSAGraph>>* graphes);

 private:
  // std::vector<std::unique_ptr<mir::SSAGraph>> graphs_;
  std::vector<Place> valid_places_;
  Scope* exec_scope_{};

  std::vector<mir::Pass*> passes_;

  std::vector<std::unique_ptr<mir::SSAGraph>> graphs_;

  core::KernelPickFactor kernel_pick_factor_;
};

//! The default optimizer.
std::unique_ptr<RuntimeProgram> RunDefaultOptimizer(
    Program&& program,
    const std::vector<Place>& valid_places,
    core::KernelPickFactor kernel_pick_factor,
    const std::vector<std::string>& passes);

}  // namespace lite
}  // namespace paddle
