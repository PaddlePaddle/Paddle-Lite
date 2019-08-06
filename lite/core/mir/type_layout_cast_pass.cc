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

#include "lite/core/mir/type_layout_cast_pass.h"
#include <list>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {

void TypeLayoutTransformPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // Start from inputs of the graph, those should have place set.
  std::list<Node*> nodes;
  for (auto& node : graph->mutable_nodes()) {
    nodes.push_back(&node);
  }

  for (auto& node : nodes) {
    if (!node->IsStmt()) continue;
    auto inlinks = node->inlinks;
    for (auto* in : inlinks) {
      ComplementInputs(graph.get(), node, in);
    }
  }
  VLOG(3) << "\n" << Visualize(graph.get());
}

void TypeLayoutTransformPass::ComplementInputs(SSAGraph* graph,
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
  if (!DataLayoutCompatibleTo(*in->AsArg().type, *decl_arg_type)) {
    LOG(INFO) << "found Layout unmatched tensor: " << in->AsArg().name
              << " for kernel " << inst.op()->DebugString() << " "
              << *in->AsArg().type << " -> " << *decl_arg_type;
    AddLayoutInst(*in->AsArg().type,
                  *decl_arg_type,
                  in,
                  graph,
                  inst_node,
                  graph->valid_places());
  }
}

void TypeLayoutTransformPass::AddLayoutInst(
    const Type& from,
    const Type& to,
    Node* in,
    SSAGraph* graph,
    Node* inst_node,
    const std::vector<Place>& valid_places) {
  CHECK(!valid_places.empty()) << "valid_place should be set";

  CHECK(in->IsArg());
  auto node_id = [&] { return graph->nodes().size(); };
  auto layout_output_name =
      string_format("%s/trans/%d", in->AsArg().name.c_str(), node_id());
  auto* layout_output_arg = graph->NewArgumentNode(layout_output_name);
  auto* layout_inst = graph->NewInstructNode();

  std::string layout_type = "layout";
  // create Op and kernels.
  auto layout_op = LiteOpRegistry::Global().Create(layout_type);
  CHECK(layout_op) << "create op [" << layout_op << "] failed";
  // CHECK(layout_op);
  // Create the new var manually.
  inst_node->AsStmt().op()->scope()->Var(layout_output_name);

  // Create IoCopy Instruction.
  cpp::OpDesc op_desc;
  op_desc.SetType(layout_type);
  op_desc.SetInput("Input", {in->AsArg().name});
  op_desc.SetOutput("Out", {layout_output_name});

  layout_op->Attach(op_desc, inst_node->AsStmt().op()->scope());
  auto kernels = layout_op->CreateKernels(valid_places);
  LOG(INFO) << "in pass add_layout: layout create kernels " << kernels.size();
  std::vector<std::unique_ptr<KernelBase>> selected_kernels;
  // fix(MyPandaShaoxiang): select kernel that input_dcl_type same as in.type
  bool is_found = false;
  for (auto& kernel : kernels) {
    const Type* in_arg_ty = kernel->GetInputDeclType("Input");
    const Type* out_arg_ty = kernel->GetOutputDeclType("Out");
    if (in_arg_ty->precision() == from.precision() &&
        in_arg_ty->target() == from.target() &&
        in_arg_ty->layout() == from.layout()) {
      is_found = true;
      selected_kernels.emplace_back(std::move(kernel));
      // we pick the kernel
      layout_inst->AsStmt(layout_type, std::move(kernels), layout_op);
      break;
    }
  }
  CHECK(is_found) << "Can't find a layout  kernel for layout op: " << from
                  << ":" << in->AsArg().name << "->" << to << ":"
                  << inst_node->AsStmt().op_info()->Type();

  // Remove the old link
  RemoveDirectedLink(in, inst_node);

  // Update the original instruction OpDesc.
  // Update its input to the layout_output_name
  // Add new link, var -> new_inst, new_inst->newarg, newarg->inst
  DirectedLink(in, layout_inst);
  DirectedLink(layout_inst, layout_output_arg);
  DirectedLink(layout_output_arg, inst_node);

  // reset opdesc and update kernel information
  UpdateInputTo(inst_node->AsStmt().op()->mutable_op_info(),
                in->AsArg().name,
                layout_output_name);
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

  std::string tmp;
  if (inst_node->AsStmt().op_info()->GetInputArgname("a", &tmp)) {
    CHECK(false) << "get old a " << tmp;
  }

  for (auto& kernel : inst_node->AsStmt().kernels()) {
    LOG(INFO) << "kernel info: " << kernel->name();
    inst_node->AsStmt().op()->AttachKernel(kernel.get());
  }

  graph->CheckValid();
}

void TypeLayoutTransformPass::SetValidPlaces(
    const std::vector<Place>& valid_places) {
  CHECK(!valid_places.empty());
  valid_places_ = valid_places;
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(type_layout_cast_pass,
                  paddle::lite::mir::TypeLayoutTransformPass);
