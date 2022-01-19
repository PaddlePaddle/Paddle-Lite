// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/optimizer/mir/keepdim_convert_pass.h"
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/optimizer/mir/graph_visualize_pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {

void KeepdimConvertPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  std::list<Node*> nodes;
  for (auto& node : graph->StmtTopologicalOrder()) {
    nodes.push_back(node);
  }

  CHECK(!valid_places_.empty());

  // record the copied node.
  std::map<std::string, Node*> copied_nodes;
  std::vector<std::string> skip_ops = {
      "while", "conditional_block", "write_back"};

  for (auto& node : nodes) {
    auto op_type = node->AsStmt().op_type();
    auto iter = std::find(skip_ops.begin(), skip_ops.end(), op_type);
    if (!node->IsStmt() || iter != skip_ops.end()) continue;
    auto inlinks = node->inlinks;
    for (auto* in : inlinks) {
      ComplementInputs(graph.get(), node, in, &copied_nodes);
    }
  }
}

void KeepdimConvertPass::ComplementInputs(
    SSAGraph* graph,
    Node* inst_node,
    Node* in,
    std::map<std::string, Node*>* copied_nodes) {
  // If this input is out of date.
  if (inst_node->inlinks.end() ==
      std::find(inst_node->inlinks.begin(), inst_node->inlinks.end(), in))
    return;

  CHECK(inst_node->IsStmt());
  auto& inst = inst_node->AsStmt();
  CHECK(in->IsRoleSet());
  CHECK(in->IsArg());
  auto in_arg_name = in->AsArg().name;
  VLOG(8) << "in_arg_name: " << in_arg_name;
  std::string tmp;
  CHECK(inst.op_info()->GetInputArgname(in_arg_name, &tmp));
  auto decl_arg_type = inst.picked_kernel().GetInputDeclType(tmp);
  CHECK(in->AsArg().type);

  auto check_attr = [](Node* p) -> bool {
    const auto op_desc = p->stmt()->op_info();
    if (op_desc->HasAttr("keepdims")) {
      if (!op_desc->GetAttr<bool>("keepdims")) return true;
    } else if (op_desc->HasAttr("keep_dim")) {
      if (!op_desc->GetAttr<bool>("keep_dim")) return true;
    }
    return false;
  };

  if (check_attr(in)) {
    VLOG(8) << "found keepdim is false: " << in_arg_name << " for kernel "
            << inst.op()->DebugString() << " " << *in->AsArg().type << " -> "
            << *decl_arg_type;
    AddReshapeInst(*in->AsArg().type,
                   *decl_arg_type,
                   in,
                   graph,
                   inst_node,
                   copied_nodes,
                   valid_places_);
  }
}

void KeepdimConvertPass::AddReshapeInst(
    const Type& from,
    const Type& to,
    Node* in,
    SSAGraph* graph,
    Node* inst_node,
    std::map<std::string, Node*>* copied_nodes,
    const std::vector<Place>& valid_places) {
  CHECK(!valid_places.empty()) << "valid_place should be set";
  CHECK(in->IsArg());

  auto reshape_output_name =
      string_format("%s/trans", in->AsArg().name.c_str());

  if (copied_nodes->count(in->AsArg().name)) {
    // Remove the old link
    RemoveDirectedLink(in, inst_node);

    // Update the original instruction OpDesc.
    // Update its input to the reshape_output_name
    // Add new link, newarg->inst
    DirectedLink(copied_nodes->at(in->AsArg().name),
                 inst_node);  // [reshape kernel]'s output -> [current kernel]

    UpdateInstNode(in, graph, inst_node, reshape_output_name);
  } else {
    auto* reshape_output_arg = graph->NewArgumentNode(reshape_output_name);
    // Set the place for reshape_output_arg node
    bool is_tensor = from.IsTensor();
    CHECK(is_tensor) << "only support tensor.";

    reshape_output_arg->AsArg().type =
        LiteType::GetTensorTy(from.target(), from.precision(), from.layout());

    auto* reshape_inst = graph->NewInstructNode();

    std::string reshape_type = "reshape";
    // create Op and kernels.
    auto reshape_op = LiteOpRegistry::Global().Create(reshape_type);
    CHECK(reshape_op) << "create op [" << reshape_op << "] failed";
    // Create the new var manually.
    inst_node->AsStmt().op()->scope()->Var(reshape_output_name);

    auto get_tensor_dims = [](const Node* in) -> DDimLite {
      std::string var_name;
      auto* inst = in->stmt();
      const auto op = inst->op();
      const auto* op_info = inst->op_info();
      auto var_names = op_info->output_names();
      CHECK_EQ(var_names.size(), 1);
      var_name = var_names[0];

      auto* scope = op->scope();
      auto* var = scope->FindVar(var_name);
      if (var == nullptr) {
        LOG(FATAL) << "var is nullptr! var_name: " << var_name;
      }
      const auto& tensor = var->Get<Tensor>();
      return tensor.dims();
    };

    // Create reshape Instruction.
    cpp::OpDesc op_desc;
    op_desc.SetType(reshape_type);
    op_desc.SetInput("X", {in->AsArg().name});
    op_desc.SetOutput("Out", {reshape_output_name});
    op_desc.SetAttr("shape", get_tensor_dims(in));

    reshape_op->Attach(op_desc, inst_node->AsStmt().op()->scope());
    auto kernels = reshape_op->CreateKernels(valid_places);
    bool is_found = false;
    std::vector<std::unique_ptr<KernelBase>> selected_kernels;
    for (auto& kernel : kernels) {
      const Type* in_arg_ty = nullptr;
      const Type* out_arg_ty = nullptr;
      in_arg_ty = kernel->GetInputDeclType("X");
      out_arg_ty = kernel->GetOutputDeclType("Out");

      VLOG(4) << "------ kernel info -------";
      VLOG(4) << "*in_arg_ty(reshape kernel input):" << *in_arg_ty;
      VLOG(4) << "from(last kernel output):" << from;
      VLOG(4) << "out_arg_ty(reshape kernel output):" << *out_arg_ty;
      VLOG(4) << "to:" << to << "\n";

      if (TypeCompatible(*in_arg_ty, from) &&
          TargetCompatibleTo(*out_arg_ty, to)) {
        VLOG(4) << "picked";
        is_found = true;
      }

      if (is_found) {
        selected_kernels.emplace_back(std::move(kernel));
        // we pick the kernel
        reshape_inst->AsStmt(
            reshape_type, std::move(selected_kernels), reshape_op);
        (*copied_nodes)[in->AsArg().name] = reshape_output_arg;
        break;
      }

      VLOG(4) << "not picked";
    }

    CHECK(is_found) << "Can't find a reshape kernel for reshape op: " << from
                    << ":" << in->AsArg().name << " -> " << to << ":"
                    << inst_node->AsStmt().op_info()->Type();
    // Remove the old link
    RemoveDirectedLink(in, inst_node);

    // Update the original instruction OpDesc.
    // Update its input to the reshape_output_name
    // Add new link, var -> new_inst, new_inst->newarg, newarg->inst
    DirectedLink(in,
                 reshape_inst);  // [last kernel]'s output -> [reshape kernel]
    DirectedLink(
        reshape_inst,
        reshape_output_arg);  // [reshape kernel] -> [reshape kernel]'s output
    DirectedLink(reshape_output_arg,
                 inst_node);  // [reshape kernel]'s output -> [current kernel]

    UpdateInstNode(in, graph, inst_node, reshape_output_name);
  }

  std::string tmp;
  if (inst_node->AsStmt().op_info()->GetInputArgname("a", &tmp)) {
    CHECK(false) << "get old a " << tmp;
  }

  for (auto& kernel : inst_node->AsStmt().kernels()) {
    VLOG(4) << "kernel info: " << kernel->name();
    inst_node->AsStmt().op()->AttachKernel(kernel.get());
  }

  graph->CheckValid();
}

void KeepdimConvertPass::SetValidPlaces(
    const std::vector<Place>& valid_places) {
  CHECK(!valid_places.empty());
  valid_places_ = valid_places;
}

void KeepdimConvertPass::UpdateInstNode(Node* in,
                                        SSAGraph* graph,
                                        Node* inst_node,
                                        std::string reshape_output_name) {
  // reset opdesc and update kernel information
  UpdateInputs(
      inst_node->AsStmt().op().get(), in->AsArg().name, reshape_output_name);
  auto original_selected_kernel =
      std::move(inst_node->AsStmt().kernels().front());
  auto update_op_info = *inst_node->AsStmt().op_info();
  // ResetOp() will change the Stmt op_info_ value,
  // after that the old op_info_ value will be nullified.
  // So, we can't pass `*inst_node->AsStmt().op_info()` into ResetOp.
  // `update_op_info` is the copy of `*inst_node->AsStmt().op_info().
  // Whenever update the op_info of a stmt, we should call its ResetOp().
  inst_node->AsStmt().ResetOp(update_op_info, graph->valid_places());
  inst_node->AsStmt().kernels().clear();
  inst_node->AsStmt().kernels().emplace_back(
      std::move(original_selected_kernel));
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(keepdim_convert_pass, paddle::lite::mir::KeepdimConvertPass)
    .BindTargets({TARGET(kOpenCL)});
