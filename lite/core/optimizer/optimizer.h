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
#include "lite/core/optimizer/mir/control_flow_op_shared_inputs_and_outputs_place_sync_pass.h"
#include "lite/core/optimizer/mir/elimination/control_flow_op_unused_inputs_and_outputs_eliminate_pass.h"
#include "lite/core/optimizer/mir/fp16_attribute_pass.h"
#include "lite/core/optimizer/mir/generate_program_pass.h"
#include "lite/core/optimizer/mir/pass_manager.h"
#include "lite/core/optimizer/mir/pass_utils.h"
#include "lite/core/optimizer/mir/post_quant_dynamic_pass.h"
#include "lite/core/optimizer/mir/ssa_graph.h"
#include "lite/core/optimizer/mir/static_kernel_pick_pass.h"
#include "lite/core/optimizer/mir/type_target_cast_pass.h"
#include "lite/core/optimizer/mir/x86_int8_attribute_pass.h"
#include "lite/core/program.h"
#include "lite/core/types.h"
#include "lite/model_parser/model_parser.h"

namespace paddle {
namespace lite {

// TODO(hong1986032) Support the following passes for the subblocks
const std::set<std::string> kSubblockUnsupportedPasses(
    {"memory_optimize_pass", "xpu_memory_optimize_pass"});

const std::set<std::string> kSubblockSkippedPasses(
    {"fill_constant_calc_offline_pass",
     "scale_calc_offline_pass",
     "unsqueeze_calc_offline_pass",
     "range_calc_offline_pass",
     "assign_value_calc_offline_pass",
     "ssd_boxes_calc_offline_pass",
     "p_norm_fill_constant_max_div_fuse_pass"});

/*
 * lite::Optimizer optimize a program. It utilize the mir passes to analysis the
 * program and export an optimized program.
 * Example :
 *       // (1) Create an optimizer
 *       Optimizer optim(valid_places, kernel_pick_factor);
 *       // (2) add an optimizer method
 *       optim.AddPass("post_quant_dynamic_pass");
 *       // (3) analysis a program to export an optimized program
 *       auto program_ = optim.Run(std::move(program));
 */
class Optimizer {
 public:
  Optimizer(const std::vector<Place>& valid_places,
            core::KernelPickFactor kernel_pick_factor)
      : valid_places_(valid_places), kernel_pick_factor_(kernel_pick_factor) {
    CHECK(!valid_places.empty()) << "At least one valid_place should be set";
  }

  // Append a pass to the optimizer.
  void AddPass(const std::string& pass_name);
  // Optimize a program to generate a runtime program.
  std::unique_ptr<RuntimeProgram> Run(Program&& program);

 protected:
  // Run all the added passes.
  void ApplyPasses(std::vector<std::unique_ptr<mir::SSAGraph>>* graphes);

  // Generate the optimized runtime program.
  std::unique_ptr<RuntimeProgram> GenRuntimeProgram(
      std::vector<std::unique_ptr<mir::SSAGraph>>* graphs);

  void InitTargetTypeTransformPass();
  void InitControlFlowOpUnusedInputsAndOutputsEliminatePass();
  void InitControlFlowOpSharedInputsAndOutputsPlaceSyncPass();
  void SpecifyKernelPickTactic(core::KernelPickFactor factor);
  Scope* exec_scope() { return exec_scope_; }

 private:
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
    const std::vector<std::string>& passes,
    const lite_api::CxxConfig& config);
}  // namespace lite
}  // namespace paddle
