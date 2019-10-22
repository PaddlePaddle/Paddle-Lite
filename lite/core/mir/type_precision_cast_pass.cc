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

#include "lite/core/mir/type_precision_cast_pass.h"
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void PrecisionCastPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // Start from inputs of the graph, those should have place set.
  std::list<Node*> nodes;
  for (auto& node : graph->StmtTopologicalOrder()) {
    nodes.push_back(node);
  }

  for (auto& node : nodes) {
    if (!node->IsStmt() || node->AsStmt().op_type() == "while") continue;
    auto inlinks = node->inlinks;
    for (auto* in : inlinks) {
      ComplementInputs(graph.get(), node, in);
    }
  }
}

void PrecisionCastPass::ComplementInputs(SSAGraph* graph,
                                         Node* inst_node,
                                         Node* in) {
  // If this input is out of date.
  if (inst_node->inlinks.end() ==
      std::find(inst_node->inlinks.begin(), inst_node->inlinks.end(), in))
    return;

  CHECK(inst_node->IsStmt());
  auto& inst = inst_node->AsStmt();
  CHECK(in->IsRoleSet());
  CHECK(in->IsArg());
  auto in_arg_name = in->AsArg().name;
  std::string tmp;
  CHECK(inst.op_info()->GetInputArgname(in_arg_name, &tmp));
  auto decl_arg_type = inst.picked_kernel().GetInputDeclType(tmp);
  CHECK(in->AsArg().type);
  VLOG(4) << inst.picked_kernel().name();
  // if (!in->AsArg().is_weight && !PrecisionCompatibleTo(*in->AsArg().type,
  // *decl_arg_type)) {
  if (!PrecisionCompatibleTo(*in->AsArg().type, *decl_arg_type)) {
    VLOG(4) << "found Target unmatched tensor: " << in->AsArg().name
            << " for kernel " << inst.op()->DebugString() << " "
            << *in->AsArg().type << " -> " << *decl_arg_type;
    // Add an Cast instruction to make the input compatible with other dist.
    AddCastInst(*in->AsArg().type,
                *decl_arg_type,
                in,
                graph,
                inst_node,
                graph->valid_places());
  }
}

void PrecisionCastPass::AddCastInst(const Type& from,
                                    const Type& to,
                                    Node* in,
                                    SSAGraph* graph,
                                    Node* inst_node,
                                    const std::vector<Place>& valid_places) {
  CHECK(!valid_places.empty()) << "valid_place should be set";

  // var -> new_transform_op -> new_var -> inst
  // So there will be a new Argument node and a new Cast Statement Node.
  CHECK(in->IsArg());
  // auto node_id = [&] { return graph->nodes().size(); };
  auto cast_op_output_name = in->AsArg().name + "/precision_trans";
  // in->AsArg().name + "/precision_trans/" + std::to_string(node_id());
  auto* cast_op_output_arg = graph->NewArgumentNode(cast_op_output_name);
  cast_op_output_arg->AsArg().type =
      LiteType::GetTensorTy(from.target(), to.precision(), from.layout());
  auto* cast_inst = graph->NewInstructNode();

  // create Op and kernels.
  bool in_persist = in->AsArg().is_weight || in->AsArg().is_persist;
  std::string cast_type = in_persist ? "calib_once" : "calib";
  cast_op_output_arg->AsArg().is_persist = in_persist;
  auto cast_op = LiteOpRegistry::Global().Create(cast_type);
  CHECK(cast_op) << "create op [" << cast_op << "] failed";

  // Create the new var manually.
  inst_node->AsStmt().op()->scope()->Var(cast_op_output_name);

  // Create Calib Instruction.
  cpp::OpDesc op_desc;
  op_desc.SetType(cast_type);
  op_desc.SetInput("Input", {in->AsArg().name});
  op_desc.SetOutput("Out", {cast_op_output_name});
  if (inst_node->AsStmt().op_info()->HasAttr("input_scale")) {
    op_desc.SetAttr(
        "scale", inst_node->AsStmt().op_info()->GetAttr<float>("input_scale"));
  }
  cast_op->Attach(op_desc, inst_node->AsStmt().op()->scope());
  auto kernels = cast_op->CreateKernels(valid_places);
  std::vector<std::unique_ptr<KernelBase>> selected_kernels;
  bool is_found = false;
  for (auto& kernel : kernels) {
    const Type* in_arg_ty = kernel->GetInputDeclType("Input");
    const Type* out_arg_ty = kernel->GetOutputDeclType("Out");
    if (TypeCompatible(*in_arg_ty, from) &&
        out_arg_ty->precision() == to.precision()) {
      is_found = true;
      selected_kernels.emplace_back(std::move(kernel));
      // we pick the kernel
      cast_inst->AsStmt(cast_type, std::move(selected_kernels), cast_op);
      break;
    }
  }

  CHECK(is_found) << "Can't find a Cast kernel for Cast op: " << from << ":"
                  << in->AsArg().name << "->" << to << ":"
                  << inst_node->AsStmt().op_info()->Type();

  // Remove the old link
  RemoveDirectedLink(in, inst_node);

  // Update the original instruction OpDesc.
  // Update its input to the io_copy_output_name

  // Add new link, var -> new_inst, new_inst->newarg, newarg->inst
  DirectedLink(in, cast_inst);
  DirectedLink(cast_inst, cast_op_output_arg);
  DirectedLink(cast_op_output_arg, inst_node);

  // reset opdesc and update kernel information
  UpdateInputTo(inst_node->AsStmt().op()->mutable_op_info(),
                in->AsArg().name,
                cast_op_output_name);

  // recreate the op
  auto original_selected_kernel =
      std::move(inst_node->AsStmt().kernels().front());
  auto updated_op_info = *inst_node->AsStmt().mutable_op_info();

  inst_node->AsStmt().ResetOp(updated_op_info, graph->valid_places());
  inst_node->AsStmt().kernels().clear();
  inst_node->AsStmt().kernels().emplace_back(
      std::move(original_selected_kernel));
  for (auto& kernel : inst_node->AsStmt().kernels()) {
    VLOG(4) << "kernel info: " << kernel->name();
    inst_node->AsStmt().op()->AttachKernel(kernel.get());
  }
  graph->CheckValid();
}

void PrecisionCastPass::SetValidPlaces(const std::vector<Place>& valid_places) {
  CHECK(!valid_places.empty());
  valid_places_ = valid_places;
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(type_precision_cast_pass,
                  paddle::lite::mir::PrecisionCastPass)
    .BindTargets({TARGET(kAny)})
    .BindKernel("calib_once")
    .BindKernel("calib");
