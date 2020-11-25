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
#include "lite/core/mir/type_precision_cast_pass.h"
#include "lite/operators/subgraph_op.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {

void TypeLayoutTransformPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // Start from inputs of the graph, those should have place set.
  VLOG(4) << "\n" << Visualize(graph.get());
  std::list<Node*> nodes;
  for (auto& node : graph->StmtTopologicalOrder()) {
    nodes.push_back(node);
  }

  VLOG(4) << "nodes.size():" << nodes.size();
  for (auto& node : nodes) {
    VLOG(4) << "!node->IsStmt():" << !node->IsStmt();
    if (!node->IsStmt() || node->AsStmt().op_type() == "while") continue;
    auto inlinks = node->inlinks;
    VLOG(4) << "============== node->AsStmt().op_type():"
            << node->AsStmt().op_type() << " inlinks.size():" << inlinks.size()
            << " ================";
    for (auto* in : inlinks) {
      ComplementInputs(graph.get(), node, in);
    }
  }
  VLOG(4) << "\n" << Visualize(graph.get());
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
  VLOG(4) << "found Target tensor: " << in->AsArg().name;
  CHECK(in->IsRoleSet());
  CHECK(in->IsArg());
  auto in_arg_name = in->AsArg().name;
  std::string inst_in_tensor_name;
  CHECK(inst.op_info()->GetInputArgname(in_arg_name, &inst_in_tensor_name));
  auto decl_arg_type =
      inst.picked_kernel().GetInputDeclType(inst_in_tensor_name);

  CHECK(in->AsArg().type);
  VLOG(3) << "\n inst_in_tensor_name:" << inst_in_tensor_name
          << "\n in->AsArg().name:" << in->AsArg().name
          << "\n *in->AsArg().type:" << *in->AsArg().type
          << "\n *decl_arg_type:" << *decl_arg_type
          << "\n inst.op()->DebugString():" << inst.op()->DebugString();

  // TODO(ysh329): conflict if tensor with kARM target but kImageDefault(OpenCL
  // layout).
  // not a good judge, but don't find the source of this issue from
  // static_pick_kernel_pass
  // to this pass.
  auto is_host = [](TargetType x) -> bool {
    return x == TARGET(kHost) || x == TARGET(kX86) || x == TARGET(kARM);
  };
  auto* in_arg_type = const_cast<Type*>(in->AsArg().type);
  if (is_host(in_arg_type->target()) &&
      in_arg_type->layout() == DATALAYOUT(kImageDefault)) {
    return;
  }

  if (!DataLayoutCompatible(*in->AsArg().type, *decl_arg_type)) {
    VLOG(4) << "found Layout unmatched tensor: " << in->AsArg().name
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
  // auto node_id = [&] { return graph->nodes().size(); };
  auto layout_output_name =
      string_format("%s/layout_trans", in->AsArg().name.c_str());
  auto* layout_output_arg = graph->NewArgumentNode(layout_output_name);
  layout_output_arg->AsArg().type =
      LiteType::GetTensorTy(from.target(), from.precision(), to.layout());

  auto* layout_inst = graph->NewInstructNode();

  bool in_persist = in->AsArg().is_weight || in->AsArg().is_persist;
  std::string layout_type = in_persist ? "layout_once" : "layout";
  // create Op and kernels.
  auto layout_op = LiteOpRegistry::Global().Create(layout_type);
  CHECK(layout_op) << "create op [" << layout_op << "] failed";
  layout_output_arg->AsArg().is_persist = in_persist;
  // Create the new var manually.
  inst_node->AsStmt().op()->scope()->Var(layout_output_name);

  // Create IoCopy Instruction.
  cpp::OpDesc op_desc;
  op_desc.SetType(layout_type);
  op_desc.SetInput("Input", {in->AsArg().name});
  op_desc.SetOutput("Out", {layout_output_name});

  layout_op->Attach(op_desc, inst_node->AsStmt().op()->scope());
  auto kernels = layout_op->CreateKernels(valid_places);
  std::vector<std::unique_ptr<KernelBase>> selected_kernels;
  bool is_found = false;
  for (auto& kernel : kernels) {
    const Type* in_arg_ty = kernel->GetInputDeclType("Input");
    const Type* out_arg_ty = kernel->GetOutputDeclType("Out");

    // layout kernel choose
    //   must ignore [layout check] for layout of kernels's input and output
    // note: replace LITE_WITH_OPENCL macro with judge input and output target
    // of layout_trans
    if ((in_arg_ty->target() == TARGET(kOpenCL) ||
         out_arg_ty->target() == TARGET(kOpenCL)) &&  // judge OpenCL first
        (TargetCompatibleTo(*in_arg_ty, from) &&
         /* skip precision check: PrecisionCompatibleTo(*in_arg_ty, from) &&*/
         DeviceCompatibleTo(*in_arg_ty, from) &&
         out_arg_ty->layout() == to.layout())) {
      is_found = true;
    } else if (TypeCompatible(*in_arg_ty, from) &&
               out_arg_ty->layout() == to.layout()) {
      is_found = true;
    }
    if (is_found) {
      selected_kernels.emplace_back(std::move(kernel));
      // we pick the kernel
      layout_inst->AsStmt(layout_type, std::move(selected_kernels), layout_op);
      break;
    }
  }

  CHECK(is_found) << "Can't find a layout kernel for layout op: " << from << ":"
                  << in->AsArg().name << "->" << to << ":"
                  << inst_node->AsStmt().op_info()->Type();
  VLOG(4) << "========= final picked layout kernel ========= ";
  VLOG(4) << "[info]:" << layout_inst->AsStmt().picked_kernel().name();
  VLOG(4) << "[summary]:" << layout_inst->AsStmt().picked_kernel().summary()
          << "\n";

  // Remove the old link
  RemoveDirectedLink(in, inst_node);

  // Update the original instruction OpDesc.
  // Update its input to the layout_output_name
  // Add new link, var -> new_inst, new_inst->newarg, newarg->inst
  DirectedLink(in, layout_inst);
  DirectedLink(layout_inst, layout_output_arg);
  DirectedLink(layout_output_arg, inst_node);

  // reset opdesc and update kernel information
  UpdateInputs(
      inst_node->AsStmt().op().get(), in->AsArg().name, layout_output_name);
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
    inst_node->AsStmt().op()->AttachKernel(kernel.get());
  }

  graph->CheckValid();
}

void TypeLayoutTransformPass::SetValidPlaces(
    const std::vector<Place>& valid_places) {
  CHECK(!valid_places.empty());
  valid_places_ = valid_places;
}

void OpenCLTypeLayoutTransformPass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  // Start from inputs of the graph, those should have place set.
  VLOG(4) << "\n" << Visualize(graph.get());
  std::list<Node*> nodes;
  for (auto& node : graph->StmtTopologicalOrder()) {
    nodes.push_back(node);
  }

  VLOG(4) << "nodes.size():" << nodes.size();
  for (auto& node : nodes) {
    VLOG(4) << "!node->IsStmt():" << !node->IsStmt();
    if (!node->IsStmt() || node->AsStmt().op_type() == "while") continue;
    VLOG(1) << "node->AsStmt().op_type():" << node->AsStmt().op_type();
    if (node->AsStmt().op_type() == "layout" ||
        node->AsStmt().op_type() == "io_copy") {
      auto new_op = node->AsStmt().mutable_op_info();
      int process_type = 1;
      new_op->SetAttr("process_type", process_type);
    }
  }
  VLOG(4) << "\n" << Visualize(graph.get());
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(type_layout_cast_pass,
                  paddle::lite::mir::TypeLayoutTransformPass)
    .BindTargets({TARGET(kAny)})
    .ExcludeTargets({TARGET(kMLU)})
    .BindKernel("layout_once")
    .BindKernel("layout");

REGISTER_MIR_PASS(type_layout_cast_preprocess_pass,
                  paddle::lite::mir::OpenCLTypeLayoutTransformPass)
    .BindTargets({TARGET(kAny)})
    .ExcludeTargets({TARGET(kMLU)})
    .BindKernel("layout_once")
    .BindKernel("layout");
