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

#include "lite/core/mir/elimination/remove_tf_redundant_ops_pass.h"
#include <set>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher.h"
#include "lite/model_parser/cpp_desc.h"

namespace paddle {
namespace lite {
namespace mir {

void RemoveTFRedundantOpsPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  RemoveSqueeze2Reshape2Pattern(graph);
  RemoveReshape2Pattern(graph);
}

void RemoveTFRedundantOpsPass::RemoveReshape2Pattern(
    const std::unique_ptr<SSAGraph>& graph) {
  bool found = false;
  Node* softmax_node{nullptr};
  Node* reshape2_node{nullptr};
  std::string reshape2_out_arg_name;
  Node* fetch_node{nullptr};
  std::string fetch_in_arg_name;
  DDim softmax_out_dims;
  DDim reshape2_out_dims;

  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (op_node->AsStmt().picked_kernel().op_type() == "softmax") {
      softmax_node = op_node;
    } else if (op_node->AsStmt().picked_kernel().op_type() == "reshape2") {
      reshape2_node = op_node;
    } else if (op_node->AsStmt().picked_kernel().op_type() == "fetch") {
      fetch_node = op_node;
      fetch_in_arg_name = fetch_node->inlinks.front()->AsArg().name;
    }
  }

  if (softmax_node == nullptr || reshape2_node == nullptr) {
    return;
  }

  // Get out tensor dims of softmax, reshape2
  auto* scope = softmax_node->AsStmt().op()->scope();
  auto softmax_out_arg_name = softmax_node->outlinks.front()->AsArg().name;
  auto softmax_out_tensor =
      scope->FindVar(softmax_out_arg_name)->Get<lite::Tensor>();
  softmax_out_dims = softmax_out_tensor.dims();

  for (auto out_node : reshape2_node->outlinks) {
    if (out_node->IsArg() && out_node->outlinks.size() != 0) {
      reshape2_out_arg_name = reshape2_node->outlinks.front()->AsArg().name;
      auto reshape2_out_tensor =
          scope->FindVar(reshape2_out_arg_name)->Get<lite::Tensor>();
      reshape2_out_dims = reshape2_out_tensor.dims();
    }
  }

  VLOG(3) << "reshape2_out_dims:" << reshape2_out_dims;
  VLOG(3) << "softmax_out_dims:" << softmax_out_dims;
  VLOG(3) << "found:" << found;

  if (softmax_out_dims == reshape2_out_dims &&
      softmax_node->outlinks.front() == reshape2_node->inlinks.front() &&
      reshape2_out_arg_name == fetch_in_arg_name) {
    found = true;
  }

  if (found) {
    // link out_arg to op
    IR_NODE_LINK_TO(softmax_node->outlinks.front(), fetch_node);

    // collect nodes to safe remove
    std::set<const Node*> nodes_to_remove;
    auto remove_inst_node_and_out_args_node = [&](Node* n) {
      nodes_to_remove.insert(n);
      for (auto& out : n->outlinks) {
        nodes_to_remove.insert(out);
      }
    };

    remove_inst_node_and_out_args_node(reshape2_node);
    GraphSafeRemoveNodes(graph.get(), nodes_to_remove);
    auto fetch_op_desc = fetch_node->AsStmt().mutable_op_info();
    fetch_op_desc->SetInput("X",
                            {softmax_node->outlinks.front()->AsArg().name});
  }
  VLOG(5) << "\n" << Visualize(graph.get());
}

void RemoveTFRedundantOpsPass::RemoveSqueeze2Reshape2Pattern(
    const std::unique_ptr<SSAGraph>& graph) {
  VLOG(5) << Visualize(graph.get());
  bool found = false;

  // find out_arg->squeeze2
  // find out_arg_dims of out_arg
  Node* out_arg_node{nullptr};
  DDim out_arg_dims;
  Node* squeeze2_node{nullptr};

  // find squeeze2->reshape2
  // find output dims of squeeze2 and reshape2 nodes
  DDim squeeze2_out_dims;
  Node* reshape2_node{nullptr};
  Node* reshape2_out_node{nullptr};
  DDim reshape2_out_dims;

  // find next inst node of reshape2
  Node* next_inst_node_of_reshape2_out{nullptr};

  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->AsStmt().picked_kernel().op_type() != "squeeze2") continue;
    auto* scope = node->AsStmt().op()->scope();

    // find inlinks of squeeze2: out_arg_node
    squeeze2_node = node;
    auto squeeze2_inlinks = squeeze2_node->inlinks;
    VLOG(5) << "squeeze2_inlinks.size():" << squeeze2_inlinks.size();
    for (auto& in_link : squeeze2_inlinks) {
      if (in_link->IsArg() && squeeze2_inlinks.size() == 1) {
        out_arg_node = in_link;
        auto* var = scope->FindVar(out_arg_node->AsArg().name);
        out_arg_dims = var->Get<lite::Tensor>().dims();
        VLOG(5) << "arg name:" << out_arg_node->AsArg().name
                << " dims:" << out_arg_dims;
      } else {
        // found mutli-input links
        continue;
      }
    }

    // find squeeze2->reshape2 pattern
    // and output dims of squeeze2, reshape2 nodes
    auto squeeze2_outlinks = squeeze2_node->outlinks;
    for (auto& squeeze2_out_link : squeeze2_outlinks) {
      if (squeeze2_out_link->IsArg() &&
          squeeze2_out_link->outlinks.size() != 0) {
        auto* squeeze2_out_var =
            scope->FindVar(squeeze2_out_link->AsArg().name);
        squeeze2_out_dims = squeeze2_out_var->Get<lite::Tensor>().dims();

        VLOG(5) << "squeeze2_out_arg.name:" << squeeze2_out_link->AsArg().name
                << " squeeze2_out_dims:" << squeeze2_out_dims
                << " squeeze2_out_link->outlinks.size():"
                << squeeze2_out_link->outlinks.size();

        for (auto& out2_link : squeeze2_out_link->outlinks) {
          if (out2_link->IsStmt() &&
              out2_link->AsStmt().picked_kernel().op_type() == "reshape2") {
            reshape2_node = out2_link;
            for (auto& reshape2_out_link : reshape2_node->outlinks) {
              if (reshape2_out_link->IsArg() &&
                  reshape2_out_link->outlinks.size() != 0) {
                reshape2_out_node = reshape2_out_link;
                auto* reshape2_out_var =
                    scope->FindVar(reshape2_out_link->AsArg().name);
                reshape2_out_dims =
                    reshape2_out_var->Get<lite::Tensor>().dims();

                VLOG(5) << "reshape2_out_node:" << reshape2_out_node
                        << " reshape2_out_name:"
                        << reshape2_out_link->AsArg().name
                        << " reshape2_out_dims:" << reshape2_out_dims;
              }
            }
          }
        }
      }
    }

    if (nullptr == reshape2_out_node) {
      VLOG(5) << "reshape2_out_node doesn't found, skip now";
      return;
    }

    // find next inst node of reshape2
    VLOG(5) << "reshape2_out_node->outlinks.size():"
            << reshape2_out_node->outlinks.size()
            << " reshape2_out_node->IsStmt():" << reshape2_out_node->IsStmt();
    VLOG(5) << "reshape2_out_node->AsArg().name:"
            << reshape2_out_node->AsArg().name;
    if (reshape2_out_node != nullptr &&
        reshape2_out_node->outlinks.size() == 1 &&
        reshape2_out_node->outlinks.front()->IsStmt()) {
      next_inst_node_of_reshape2_out = reshape2_out_node->outlinks.front();
      found = true;
      break;
      VLOG(5)
          << "next_inst_node_of_reshape2_out->picked_kernel().op_type():"
          << next_inst_node_of_reshape2_out->AsStmt().picked_kernel().op_type();
    }

    VLOG(5) << "==============================";
    VLOG(5) << "out_arg_dims:" << out_arg_dims;
    VLOG(5) << "squeeze2_out_dims:" << squeeze2_out_dims;
    VLOG(5) << "reshape2_out_dims:" << reshape2_out_dims;
    VLOG(5) << "==============================";
  }

  // replace pattern
  if (found && out_arg_dims[1] == squeeze2_out_dims[1] &&
      out_arg_dims[1] == reshape2_out_dims[1] && out_arg_dims[1] == 1001 &&
      out_arg_dims[2] == out_arg_dims[3] && out_arg_dims[2] == 1 &&
      next_inst_node_of_reshape2_out->AsStmt().picked_kernel().op_type() ==
          "softmax") {
    // link out_arg to op
    IR_NODE_LINK_TO(out_arg_node, next_inst_node_of_reshape2_out);

    // collect nodes to safe remove
    std::set<const Node*> nodes_to_remove;
    auto remove_inst_node_and_out_args_node = [&](Node* n) {
      nodes_to_remove.insert(n);
      for (auto& out : n->outlinks) {
        nodes_to_remove.insert(out);
      }
    };
    remove_inst_node_and_out_args_node(squeeze2_node);
    remove_inst_node_and_out_args_node(reshape2_node);
    GraphSafeRemoveNodes(graph.get(), nodes_to_remove);
    auto next_inst_op_desc =
        next_inst_node_of_reshape2_out->AsStmt().mutable_op_info();
    next_inst_op_desc->SetInput("X", {out_arg_node->AsArg().name});
    VLOG(5) << Visualize(graph.get());
  }
  VLOG(5) << "replace pattern fininshed";
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(remove_tf_redundant_ops_pass,
                  paddle::lite::mir::RemoveTFRedundantOpsPass)
    .BindTargets({TARGET(kOpenCL), TARGET(kARM)});
