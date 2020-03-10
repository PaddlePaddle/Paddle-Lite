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

#include "lite/core/mir/pass.h"
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

class SubgraphCastDisplayPass : public DebugPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    VLOG(4) << "== SubgraphOp Debug Info ==";
    for (auto& node : graph->mutable_nodes()) {
      if (node.IsStmt() && node.AsStmt().op_type() == "subgraph") {
        VLOG(4) << "FOUND SUBGRAPH OP";
        display_debug_info(node, "subgraph");
        break;
      }
    }
    VLOG(4) << "---------------------";
  }

  void display_debug_info(const Node& node,
                          std::string op_type,
                          bool display_in_nodes = true,
                          bool display_out_nodes = true) {
    CHECK(node.IsStmt());
    // VLOG(4) << node.AsStmt();
    if (display_in_nodes) {
      for (auto p_in_arg_node : node.inlinks) {
        CHECK(p_in_arg_node->IsArg());
        VLOG(4) << "* ARG[IN] " << p_in_arg_node->AsArg().name
                << " type: " << *p_in_arg_node->AsArg().type
                << " is_weight: " << p_in_arg_node->AsArg().is_weight
                << " is_persist: " << p_in_arg_node->AsArg().is_persist
                << " input_count: " << p_in_arg_node->inlinks.size();
        if (p_in_arg_node->inlinks.size() == 0) {
          VLOG(4) << "** END with No Op";
        }
        for (auto p_in_stmt_node : p_in_arg_node->inlinks) {
          CHECK(p_in_stmt_node->IsStmt());
          std::string stmt_op_type = p_in_stmt_node->AsStmt().op_type();
          if (stmt_op_type == "cast" || stmt_op_type == "transpose" ||
              stmt_op_type == "io_copy") {
            display_debug_info(*p_in_stmt_node, stmt_op_type, true, false);
          } else {
            VLOG(4) << "** END with op type: " << stmt_op_type;
          }
        }
      }
    }
    if (display_out_nodes) {
      for (auto p_out_arg_node : node.outlinks) {
        CHECK(p_out_arg_node->IsArg());
        VLOG(4) << "* ARG[OUT] " << p_out_arg_node->AsArg().name
                << " type: " << *p_out_arg_node->AsArg().type
                << " is_weight: " << p_out_arg_node->AsArg().is_weight
                << " is_persist: " << p_out_arg_node->AsArg().is_persist
                << " output_count: " << p_out_arg_node->outlinks.size();
        if (p_out_arg_node->outlinks.size() == 0) {
          VLOG(4) << "** END with No Op";
        }
        for (auto p_out_stmt_node : p_out_arg_node->outlinks) {
          CHECK(p_out_stmt_node->IsStmt());
          std::string stmt_op_type = p_out_stmt_node->AsStmt().op_type();
          if (stmt_op_type == "cast" || stmt_op_type == "transpose" ||
              stmt_op_type == "io_copy") {
            display_debug_info(*p_out_stmt_node, stmt_op_type, false, true);
          } else {
            VLOG(4) << "** END with op type: " << stmt_op_type;
          }
        }
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(subgraph_cast_display_pass,
                  paddle::lite::mir::SubgraphCastDisplayPass)
    .BindTargets({TARGET(kMLU)});
