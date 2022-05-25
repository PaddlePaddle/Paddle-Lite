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

#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

class FeedOpVariableTypeRecordPass : public StmtPass {
 public:
  FeedOpVariableTypeRecordPass() {}

  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    for (auto& node : graph->StmtTopologicalOrder()) {
      if (!node->IsStmt()) continue;
      auto op_info = node->AsStmt().mutable_op_info();
      auto op_type = op_info->Type();
      if (op_type != "feed") continue;
      // Initialize the type of the output variable of feed op
      CHECK_EQ(node->outlinks.size(), 1);
      auto out_var_node = node->outlinks.front();
      CHECK(out_var_node->IsArg());
      auto out_var_name = out_var_node->AsArg().name;
      auto out_var_place = out_var_node->AsArg().type->place();
      // Identify the type of the variable from the registration information of
      // the consumer op
      for (auto out_op_node : out_var_node->outlinks) {
        CHECK(out_op_node->IsStmt());
        auto out_op_info = out_op_node->AsStmt().mutable_op_info();
        auto& out_op_kernel = out_op_node->AsStmt().picked_kernel();
        std::string in_arg_name;
        CHECK(out_op_info->GetInputArgname(out_var_name, &in_arg_name));
        auto in_var_place =
            out_op_kernel.GetInputDeclType(in_arg_name)->place();
        if (in_var_place.target != TARGET(kUnk) &&
            in_var_place.target != TARGET(kAny)) {
          out_var_place.target = in_var_place.target;
        }
        if (in_var_place.precision != PRECISION(kUnk) &&
            in_var_place.precision != PRECISION(kAny)) {
          out_var_place.precision = in_var_place.precision;
        }
        if (in_var_place.layout != DATALAYOUT(kUnk) &&
            in_var_place.layout != DATALAYOUT(kAny)) {
          out_var_place.layout = in_var_place.layout;
        }
      }
      op_info->SetAttr<std::string>(kFeedTypeAttr, out_var_place.Serialize());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(feed_op_variable_type_record_pass,
                  paddle::lite::mir::FeedOpVariableTypeRecordPass)
    .BindTargets({TARGET(kNNAdapter)});
