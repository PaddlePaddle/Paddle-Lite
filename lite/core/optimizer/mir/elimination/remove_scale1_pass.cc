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

#include "lite/core/optimizer/mir/elimination/remove_scale1_pass.h"
#include <set>
#include "lite/core/optimizer/mir/graph_visualize_pass.h"
#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher.h"
#include "lite/model_parser/cpp_desc.h"

namespace paddle {
namespace lite {
namespace mir {

void RemoveScale1Pass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  auto check_scale1 = [](Node* p) -> bool {
    auto op_desc = p->stmt()->mutable_op_info();
    auto scale = op_desc->GetAttr<float>("scale");
    auto bias = op_desc->GetAttr<float>("bias");

    std::string activation_type = "";
    if (op_desc->HasAttr("activation_type")) {
      activation_type = op_desc->GetAttr<std::string>("activation_type");
    }
    bool fuse_relu = false;
    if (op_desc->HasAttr("fuse_relu")) {
      fuse_relu = op_desc->GetAttr<bool>("fuse_relu");
    }

    VLOG(3) << "scale:" << scale;
    VLOG(3) << "bias:" << bias;
    VLOG(3) << "activation_type:" << activation_type;
    VLOG(3) << "fuse_relu:" << fuse_relu;

    if (fuse_relu) {
      VLOG(3) << "skip. fuse_relu:" << fuse_relu;
      return false;
    }
    if ((scale > 1.0 + 1e-5) || (scale < 1.0 - 1e-5)) {
      VLOG(3) << "skip. scale != 1.0, scale:" << scale;
      return false;
    }
    if ((bias > 0.f + 1e-5) || (bias < 0.f - 1e-5)) {
      VLOG(3) << "skip. bias != 0.f, bias:" << bias;
      return false;
    }
    return true;
  };

  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (op_node->AsStmt().op_type() == "scale") {
      Node* scale_node = op_node;
      bool is_scale1 = check_scale1(scale_node);
      VLOG(2) << "is_scale1:" << is_scale1;
      if (is_scale1) {
        if (scale_node->inlinks.size() == 1 &&  // input arg of scale is 1
            scale_node->inlinks.front()->inlinks.size() ==
                1 &&  // last node of scale is 1
            scale_node->inlinks.front()
                ->inlinks.front()
                ->IsStmt() &&                    // last node of scale is stmt
            scale_node->outlinks.size() == 1 &&  //
            scale_node->outlinks.front()->outlinks.size() == 1 &&  //
            scale_node->outlinks.front()->outlinks.front()->IsStmt()) {
          VLOG(2) << "scale pattern found: last and next of scale is stmts.";
          Node* last_stmt_of_scale =
              scale_node->inlinks.front()->inlinks.front();
          Node* next_stmt_of_scale =
              scale_node->outlinks.front()->outlinks.front();
          auto next_op_desc = next_stmt_of_scale->AsStmt().mutable_op_info();
          if (next_op_desc->inputs().size() != 1) {
            VLOG(2) << "skip, next op of scale has input with size > 1.";
            return;
          }

          // link out_arg to op
          IR_NODE_LINK_TO(last_stmt_of_scale->outlinks.front(),
                          next_stmt_of_scale);

          // safe remove
          std::set<const Node*> nodes_to_remove;
          auto remove_inst_node_and_out_args_node = [&](Node* n) {
            nodes_to_remove.insert(n);
            for (auto& out : n->outlinks) {
              nodes_to_remove.insert(out);
            }
          };
          remove_inst_node_and_out_args_node(scale_node);
          GraphSafeRemoveNodes(graph.get(), nodes_to_remove);

          auto m = next_op_desc->inputs();
          std::string next_op_desc_input_arg_name;
          for (auto kv = m.begin(); kv != m.end(); kv++) {
            auto param = kv->first;
            next_op_desc_input_arg_name = param;
            std::vector<std::string> strs = kv->second;
            VLOG(2) << "======== param:" << param << "=========";
            for (size_t i = 0; i < strs.size(); ++i) {
              VLOG(2) << "strs[" << i << "]:" << strs[i];
            }
          }
          next_op_desc->SetInput(
              next_op_desc_input_arg_name,
              {last_stmt_of_scale->outlinks.front()->AsArg().name});
        }
      }
    }
  }
  VLOG(6) << Visualize(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(remove_scale1_pass, paddle::lite::mir::RemoveScale1Pass)
    .BindTargets({TARGET(kOpenCL)});
