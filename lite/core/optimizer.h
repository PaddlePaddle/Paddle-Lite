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
#include <memory>
#include <string>
#include <vector>
#include "lite/core/mir/generate_program_pass.h"
#include "lite/core/mir/pass_manager.h"
#include "lite/core/mir/ssa_graph.h"
#include "lite/core/mir/static_kernel_pick_pass.h"
#include "lite/core/mir/type_target_cast_pass.h"
#include "lite/core/program.h"
#include "lite/core/types.h"
#include "lite/model_parser/model_parser.h"
#ifdef LITE_WITH_NPU
#include "lite/core/mir/subgraph/generate_npu_program_pass.h"
#endif

namespace paddle {
namespace lite {

/*
 * lite::Optimizer optimize a program. It utilize the mir passes to analysis the
 * program and export an optimized program.
 */
class Optimizer {
 public:
  void Run(Program&& program,
           const std::vector<Place>& valid_places,
           core::KernelPickFactor kernel_pick_factor,
           const std::vector<std::string>& passes = {}) {
    program_ = &program;
    valid_places_ = valid_places;
    CHECK(!valid_places.empty()) << "At least one valid_place should be set";
    CHECK(!graph_) << "duplicate optimize found";
    graph_.reset(new mir::SSAGraph);
    graph_->Build(program, valid_places);
    graph_->SetValidPlaces(valid_places);

    SpecifyKernelPickTactic(kernel_pick_factor);
    InitTargetTypeTransformPass();

    if (passes.empty()) {
      RunPasses(std::vector<std::string>{
          {"lite_quant_dequant_fuse_pass",  //
           "lite_conv_bn_fuse_pass",        //
// This pass is disabled to force some opencl kernels selected for final
// running, otherwise, they will be fused to ARM fusion kernels, and the OpenCL
// devices will be discarded.
// TODO(Superjomn) Refine the fusion related design to select fusion kernels for
// devices automatically.
#ifndef LITE_WITH_OPENCL
           "lite_conv_elementwise_add_activation_fuse_pass",  //
#endif
           "lite_fc_fuse_pass",              //
           "identity_scale_eliminate_pass",  //
#ifdef LITE_WITH_LIGHT_WEIGHT_FRAMEWORK
           "lite_elementwise_add_activation_fuse_pass",  //
#endif
           "static_kernel_pick_pass",        //
           "variable_place_inference_pass",  //
           "argument_type_display_pass",     //
           "type_target_cast_pass",          //
           "variable_place_inference_pass",  //
           "argument_type_display_pass",     //
           "io_copy_kernel_pick_pass",       //
           "variable_place_inference_pass",  //
           "type_precision_cast_pass",       //
           "argument_type_display_pass",     //
           "trans_weight_pass",              //
           "runtime_context_assign_pass",    //
           "graph_visualze"}});
    } else {
      RunPasses(passes);
    }
    exec_scope_ = program.exec_scope();
  }

  void KernelPickPreferPlace(const Place& place) {
    auto* pass = mir::PassManager::Global().LookUp<mir::StaticKernelPickPass>(
        "static_kernel_pick_pass");
    CHECK(pass);
    pass->SetPreferPlace(place);
  }

  const lite::Scope* exec_scope() const { return exec_scope_; }

  // Generate a new program based on the mir graph.
  std::unique_ptr<RuntimeProgram> GenRuntimeProgram() {
    LOG(INFO) << "generate program";
    auto pass = mir::PassManager::Global().LookUp<mir::GenerateProgramPass>(
        "generate_program_pass");
    pass->Apply(graph_);
    auto program = pass->GenProgram();
    CHECK(exec_scope_);
    program->set_exec_scope(exec_scope_);
    return program;
  }

  // check the input dims in the scope, must not be empty
  void CheckInputDimsNotEmpty(const lite::Scope* scope) {
    CHECK(scope);
    auto* feed_var = scope->FindVar("feed");
    CHECK(feed_var) << "no feed variable in exec_scope: " << scope;
    auto* feed_tensor_list = feed_var->GetMutable<std::vector<lite::Tensor>>();
    CHECK_GE(feed_tensor_list->size(), 1);
    for (size_t i = 0; i < feed_tensor_list->size(); ++i) {
      CHECK(!feed_tensor_list->at(i).dims().empty())
          << "Input " << i << " dims can not be empty.";
    }
  }

  std::unique_ptr<RuntimeProgram> GenNPURuntimeProgram() {
#ifdef LITE_WITH_NPU
    CheckInputDimsNotEmpty(exec_scope_);
    auto pass = mir::PassManager::Global().LookUp<mir::GenerateNPUProgramPass>(
        "generate_npu_program_pass");
    pass->Apply(graph_);

    auto program = pass->GenProgram();
    CHECK(exec_scope_);
    program->set_exec_scope(exec_scope_);
    return program;
#else
    LOG(WARNING) << "Not compiled with NPU but use it!";
    return GenRuntimeProgram();
#endif
  }

  void InitTargetTypeTransformPass() {
    auto* pass =
        mir::PassManager::Global().LookUp<mir::TypeTargetTransformPass>(
            "type_target_cast_pass");
    CHECK(pass);
    CHECK(!valid_places_.empty());
    LOG(INFO) << "valid_places.size " << valid_places_.size();
    pass->SetValidPlaces(valid_places_);
  }

  // Generate C++ code which combines the inference program, model and weights.
  void GenCode(const std::string& code_dir);

  const mir::SSAGraph& ssa_graph() const {
    CHECK(graph_);
    return *graph_;
  }

  mir::SSAGraph* mutable_ssa_graph() {
    CHECK(graph_);
    return graph_.get();
  }

  lite::Scope* exec_scope() { return exec_scope_; }

 protected:
  void SpecifyKernelPickTactic(core::KernelPickFactor factor);

  // Specify the passes and run them.
  void RunPasses(const std::vector<std::string>& passes) {
    for (auto& x : passes) {
      LOG(INFO) << "== Running pass " << x;
      auto* pass = mir::PassManager::Global().LookUp(x);
      CHECK(pass) << "Can not find pass: " << x;
      pass->Apply(graph_);
    }
  }

 private:
  std::unique_ptr<mir::SSAGraph> graph_;
  std::vector<Place> valid_places_;
  lite::Scope* exec_scope_{};
  Program* program_{};
};

}  // namespace lite
}  // namespace paddle
