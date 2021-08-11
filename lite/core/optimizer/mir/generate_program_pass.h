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

#include <algorithm>
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/optimizer/mir/pass.h"
#include "lite/kernels/host/conditional_block_compute.h"
#include "lite/kernels/host/while_compute.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * GenerateProgramPass will build the execution program for executor from a mir
 * graph.
 */
class GenerateProgramPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

  std::unique_ptr<RuntimeProgram> GenProgram() {
    LOG(INFO) << "insts.size: " << insts_.size();

#ifdef LITE_WITH_XPU
    // generate RuntimeProgram for sub_block and set RuntimeProgram into
    // sub_block kernel
    // sub_block: while, conditional_block
    std::vector<std::string> sub_block_ops{"while", "conditional_block"};
    for (int i = static_cast<int>(insts_.size()) - 2; i >= 0; i--) {
      for (auto& inst : insts_[i]) {
        std::string op_name = inst.op()->Type();
        if (std::find(sub_block_ops.begin(), sub_block_ops.end(), op_name) ==
            sub_block_ops.end()) {
          continue;
        }

        CHECK(inst.op()->op_info()->HasAttr("sub_block"))
            << op_name << " op should have attr 'sub_block'";
        int block_idx = inst.op()->op_info()->GetAttr<int>("sub_block");
        CHECK_LT(block_idx, static_cast<int>(insts_.size()))
            << op_name
            << " op's attr 'sub_block' should be less than number of blocks.";
        std::vector<std::vector<Instruction>> sub_insts;
        sub_insts.emplace_back(std::move(insts_[block_idx]));
        std::unique_ptr<RuntimeProgram> sub_program(
            new RuntimeProgram(std::move(sub_insts)));

        if (op_name == "while") {
          auto* kernel =
              static_cast<kernels::host::WhileCompute*>(inst.mutable_kernel());
          kernel->SetRuntimeProgram(&sub_program);
        } else if (op_name == "conditional_block") {
          auto* kernel = static_cast<kernels::host::ConditionalBlockCompute*>(
              inst.mutable_kernel());
          kernel->SetRuntimeProgram(&sub_program);
        } else {
          LOG(FATAL) << "unsupported sub_block op: " << op_name;
        }
      }
    }
#endif

    // generate RuntimeProgram for main block
    std::unique_ptr<RuntimeProgram> program(
        new RuntimeProgram(std::move(insts_)));

    return program;
  }

 private:
  std::vector<std::vector<Instruction>> insts_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
