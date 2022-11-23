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
#include <set>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass.h"

namespace paddle {
namespace lite {
namespace mir {

class QuantDequantFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

  void GetInputThreshold(const std::unique_ptr<SSAGraph>& graph) {
    for (auto& node : graph->StmtTopologicalOrder()) {
      if (!node->IsStmt()) continue;

      auto& instruct = node->AsStmt();
      if (xpu_general_int8_op_types_.count(instruct.op_type()) == 0) {
        continue;
      }

      for (auto* in_node : node->inlinks) {
        CHECK(in_node->IsArg());
        if (in_node->inlinks.empty()) {
          continue;
        }

        if (!(in_node->inlinks.front()->IsStmt())) continue;
        auto& pre_op_inst = in_node->inlinks.front()->AsStmt();

        // pre link op is normal op
        if (pre_op_inst.op_info()->HasAttr("out_threshold")) {
          float pre_op_out_threshold =
              pre_op_inst.op_info()->GetAttr<float>("out_threshold");
          instruct.mutable_op_info()->SetAttr<float>("input_threshold",
                                                     pre_op_out_threshold);
        }

        // pre link op is fused conv2d/fc.
        // if (pre_op_inst.op_info()->HasAttr("Output0_scale")) {
        //   float pre_op_out_threshold =
        //       pre_op_inst.op_info()->GetAttr<std::vector<float>>(
        //           "Output0_scale")[0];
        //   instruct.mutable_op_info()->SetAttr<float>("input_threshold",
        //                                              pre_op_out_threshold);
        // }
      }
    }
  }

 private:
  std::set<std::string> xpu_general_int8_op_types_ = {"pool2d"};
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
