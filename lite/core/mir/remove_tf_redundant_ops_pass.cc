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

#include "lite/core/mir/remove_tf_redundant_ops_pass.h"
#include <set>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher.h"
#include "lite/model_parser/cpp/var_desc.h"

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
  Node* fetch_node{nullptr};
  DDim softmax_out_dims;
  DDim reshape2_out_dims;

  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (op_node->IsStmt()) {
      if (op_node->AsStmt().picked_kernel().op_type() == "softmax") {
        softmax_node = op_node;
      } else if (op_node->AsStmt().picked_kernel().op_type() == "reshape2") {
        reshape2_node = op_node;
      } else if (op_node->AsStmt().picked_kernel().op_type() == "fetch") {
        fetch_node = op_node;
      }
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
      auto reshape2_out_arg_name =
          reshape2_node->outlinks.front()->AsArg().name;
      auto reshape2_out_tensor =
          scope->FindVar(reshape2_out_arg_name)->Get<lite::Tensor>();
      reshape2_out_dims = reshape2_out_tensor.dims();
    }
  }

  LOG(INFO) << "reshape2_out_dims:" << reshape2_out_dims;
  LOG(INFO) << "softmax_out_dims:" << softmax_out_dims;
  LOG(INFO) << "found:" << found;

  if (softmax_out_dims == reshape2_out_dims) {
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
  LOG(INFO) << "start visual-- ";
  VLOG(1) << Visualize(graph.get());

  for (auto& node : graph->mutable_nodes()) {
    LOG(INFO) << "222222222222222222222";
    if (node.IsStmt())
      LOG(INFO) << "node.AsStmt().picked_kernel().op_type():"
                << node.AsStmt().picked_kernel().op_type();
    LOG(INFO) << "222222222222222222222";
    if (node.IsArg()) LOG(INFO) << "node.AsArg().name:" << node.AsArg().name;
    LOG(INFO) << "222222222222222222222";
    if (!node.IsStmt()) continue;
    LOG(INFO) << "222222222222222222222";
    auto& inst = node.AsStmt();
    LOG(INFO) << "222222222222222222222";
    auto& k = inst.picked_kernel();
    LOG(INFO) << "222222222222222222222";
    if (!(k.op_type() == "squeeze2")) continue;
    auto squeeze2 = inst.op();
    LOG(INFO) << "222222222222222222222";
    auto* scope = squeeze2->scope();

    LOG(INFO) << "\n"
              << "k->name():" << k.name() << "\n"
              << "k->op_type():" << k.op_type();

    // find out_arg->squeeze2
    // find out_arg_dims of out_arg
    Node* out_arg_node{nullptr};
    DDim out_arg_dims;
    Node* squeeze2_node{&node};

    auto squeeze2_inlinks = squeeze2_node->inlinks;
    LOG(INFO) << "squeeze2_inlinks.size():" << squeeze2_inlinks.size();
    for (auto& in_link : squeeze2_inlinks) {
      if (in_link->IsArg() && squeeze2_inlinks.size() == 1) {
        out_arg_node = in_link;
        auto arg_name = out_arg_node->AsArg().name;
        auto* var = scope->FindVar(arg_name);
        auto tensor = var->Get<lite::Tensor>();
        out_arg_dims = tensor.dims();
        LOG(INFO) << "arg name:" << arg_name << " dims:" << out_arg_dims;
      } else {
        // found mutli-input links
        return;
      }
    }

    // find squeeze2->reshape2
    // find output dims of squeeze2 and reshape2 nodes
    DDim squeeze2_out_dims;
    Node* reshape2_node{nullptr};
    Node* reshape2_out_node{nullptr};
    DDim reshape2_out_dims;

    auto squeeze2_outlinks = squeeze2_node->outlinks;
    LOG(INFO) << "outlinks.size():" << squeeze2_outlinks.size();
    for (auto& squeeze2_out_link : squeeze2_outlinks) {
      if (squeeze2_out_link->IsArg() &&
          squeeze2_out_link->outlinks.size() != 0) {
        auto& squeeze2_out_arg = squeeze2_out_link->AsArg();
        auto* squeeze2_out_var = scope->FindVar(squeeze2_out_arg.name);
        auto squeeze2_out_tensor = squeeze2_out_var->Get<lite::Tensor>();
        squeeze2_out_dims = squeeze2_out_tensor.dims();
        LOG(INFO) << "squeeze2_out_arg.name:" << squeeze2_out_arg.name
                  << " squeeze2_out_dims:" << squeeze2_out_dims;
        LOG(INFO) << "squeeze2_out_link->outlinks.size():"
                  << squeeze2_out_link->outlinks.size();
        for (auto& out2_link : squeeze2_out_link->outlinks) {
          if (out2_link->IsStmt()) {
            auto& out2_kernel = out2_link->AsStmt().picked_kernel();
            if (out2_kernel.op_type() == "reshape2") {
              reshape2_node = out2_link;
              for (auto& reshape2_out_link : reshape2_node->outlinks) {
                if (reshape2_out_link->IsArg() &&
                    reshape2_out_link->outlinks.size() != 0) {
                  reshape2_out_node = reshape2_out_link;
                  auto reshape2_out_arg_name = reshape2_out_link->AsArg().name;
                  auto* reshape2_out_var =
                      scope->FindVar(reshape2_out_arg_name);
                  auto reshape2_out_tensor =
                      reshape2_out_var->Get<lite::Tensor>();
                  reshape2_out_dims = reshape2_out_tensor.dims();
                  LOG(INFO) << "reshape2_out_var:" << reshape2_out_arg_name
                            << " reshape2_out_dims:" << reshape2_out_dims;
                }
              }
            }
          }
        }
      }
    }

    // find next inst node of reshape2
    Node* next_inst_node_of_reshape2_out{nullptr};
    LOG(INFO) << "reshape2_out_node->outlinks.size():"
              << reshape2_out_node->outlinks.size()
              << " reshape2_out_node->IsStmt():" << reshape2_out_node->IsStmt();
    LOG(INFO) << "reshape2_out_node->AsArg().name:"
              << reshape2_out_node->AsArg().name;
    if (reshape2_out_node->outlinks.size() == 1 &&
        reshape2_out_node->outlinks.front()->IsStmt()) {
      next_inst_node_of_reshape2_out = reshape2_out_node->outlinks.front();
      LOG(INFO)
          << "next_inst_node_of_reshape2_out->picked_kernel().op_type():"
          << next_inst_node_of_reshape2_out->AsStmt().picked_kernel().op_type();
    } else {
      continue;
    }

    LOG(INFO) << "==============================";
    LOG(INFO) << "out_arg_dims:" << out_arg_dims;
    LOG(INFO) << "squeeze2_out_dims:" << squeeze2_out_dims;
    LOG(INFO) << "reshape2_out_dims:" << reshape2_out_dims;
    LOG(INFO) << "==============================";

    // replace pattern
    if (out_arg_dims[1] == squeeze2_out_dims[1] &&
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
      VLOG(1) << Visualize(graph.get());
      return;
    }
    LOG(INFO) << "222222222222222222222";
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(remove_tf_redundant_ops_pass,
                  paddle::lite::mir::RemoveTFRedundantOpsPass)
    .BindTargets({TARGET(kAny)});
