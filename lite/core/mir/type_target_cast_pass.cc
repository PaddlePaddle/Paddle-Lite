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

#include "lite/core/mir/type_target_cast_pass.h"
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/mir/graph_visualize_pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/type_precision_cast_pass.h"
#include "lite/utils/string.h"

namespace paddle {
namespace lite {
namespace mir {

void TypeTargetTransformPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // Start from inputs of the graph, those should have place set.
  std::list<Node*> nodes;
  for (auto& node : graph->StmtTopologicalOrder()) {
    nodes.push_back(node);
  }

  CHECK(!valid_places_.empty());

  // record the copied node.
  std::map<std::string, Node*> copied_nodes;
  std::vector<std::string> skip_ops = {"while", "conditional_block"};

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

void TypeTargetTransformPass::ComplementInputs(
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
  VLOG(3) << "found Target tensor: " << in->AsArg().name;
  CHECK(in->IsRoleSet());
  CHECK(in->IsArg());
  auto in_arg_name = in->AsArg().name;
  std::string tmp;
  CHECK(inst.op_info()->GetInputArgname(in_arg_name, &tmp));
  auto decl_arg_type = inst.picked_kernel().GetInputDeclType(tmp);
  CHECK(in->AsArg().type);
  if (!TargetCompatibleTo(*in->AsArg().type, *decl_arg_type)) {
    VLOG(3) << "found Target unmatched tensor: " << in->AsArg().name
            << " for kernel " << inst.op()->DebugString() << " "
            << *in->AsArg().type << " -> " << *decl_arg_type;
    // Add an IoCopy instruction to make the input compatible with other dist.
    AddIoCopyInst(*in->AsArg().type,
                  *decl_arg_type,
                  in,
                  graph,
                  inst_node,
                  copied_nodes,
                  valid_places_);
  }
}

void TypeTargetTransformPass::AddIoCopyInst(
    const Type& from,
    const Type& to,
    Node* in,
    SSAGraph* graph,
    Node* inst_node,
    std::map<std::string, Node*>* copied_nodes,
    const std::vector<Place>& valid_places) {
  CHECK(!valid_places.empty()) << "valid_place should be set";
  // var -> new_transform_op -> new_var -> inst
  // So there will be a new Argument node and a new IoCopy Statement Node.

  CHECK(in->IsArg());

  // auto node_id = [&] { return graph->nodes().size(); };
  auto io_copy_output_name =
      string_format("%s/target_trans", in->AsArg().name.c_str());
  // string_format("%s/target_trans/%d", in->AsArg().name.c_str(), node_id());

  if (copied_nodes->count(in->AsArg().name)) {
    // Remove the old link
    RemoveDirectedLink(in, inst_node);

    // Update the original instruction OpDesc.
    // Update its input to the io_copy_output_name
    // Add new link, newarg->inst
    DirectedLink(copied_nodes->at(in->AsArg().name),
                 inst_node);  // [io_copy kernel]'s output -> [current kernel]

    UpdateInstNode(in, graph, inst_node, io_copy_output_name);
  } else {
    // TODO(MyPandaShaoxiang) should set same place with input?
    auto* io_copy_output_arg = graph->NewArgumentNode(io_copy_output_name);
    // Set the place for io_copy_output_arg node, the target should be equal to
    // to.target()
    // The precision and layout should be equal to from.precision(),
    // from.layout()
    io_copy_output_arg->AsArg().type =
        LiteType::GetTensorTy(to.target(), from.precision(), from.layout());
    auto* io_copy_inst = graph->NewInstructNode();

    bool in_persist = in->AsArg().is_weight || in->AsArg().is_persist;
    std::string io_copy_type = in_persist ? "io_copy_once" : "io_copy";
    io_copy_output_arg->AsArg().is_persist = in_persist;
    // create Op and kernels.
    auto io_copy_op = LiteOpRegistry::Global().Create(io_copy_type);
    CHECK(io_copy_op) << "create op [" << io_copy_op << "] failed";
    // CHECK(io_copy_op);
    // Create the new var manually.
    inst_node->AsStmt().op()->scope()->Var(io_copy_output_name);

    // Create IoCopy Instruction.
    cpp::OpDesc op_desc;
    op_desc.SetType(io_copy_type);
    op_desc.SetInput("Input", {in->AsArg().name});
    op_desc.SetOutput("Out", {io_copy_output_name});

    io_copy_op->Attach(op_desc, inst_node->AsStmt().op()->scope());
    auto kernels = io_copy_op->CreateKernels(valid_places);
    // fix(MyPandaShaoxiang): select kernel that input_dcl_type same as in.type
    bool is_found = false;
    std::vector<std::unique_ptr<KernelBase>> selected_kernels;
    for (auto& kernel : kernels) {
      const Type* in_arg_ty = kernel->GetInputDeclType("Input");
      const Type* out_arg_ty = kernel->GetOutputDeclType("Out");

      VLOG(4) << "------ kernel info -------";
      VLOG(4) << "*in_arg_ty(io_copy kernel input):" << *in_arg_ty;
      VLOG(4) << "from(last kernel output):" << from;
      VLOG(4) << "out_arg_ty(io_copy kernel output):" << *out_arg_ty;
      VLOG(4) << "to:" << to << "\n";

      // kernel choose branch for opencl backend
      //   judge inst's target whether is kOpenCL
      //   Note: to == *decl_arg_type == in of inst, not output of last inst
      // ignore [layout check] for layout between [to] and [from]
      //   Because all of origin opencl insts in model, are not default layout
      //   NCHW,
      //   so skip layout check.
      // detailed node info see below:
      //     [*in->AsArg().type] -> [from]: out of inst's previous kernel
      //     [*decl_arg_type] -> [to]: input of inst, not output of last
      //     [in_arg_ty]: in of io_copy
      //     [out_arg_ty]: out of io_copy
      //
      // noto: replace LITE_WITH_OPENCL macro with judge input and output target
      // of io_copy
      if ((in_arg_ty->target() == TARGET(kOpenCL) ||
           out_arg_ty->target() == TARGET(kOpenCL)) &&  // judge OpenCL first
          (TargetCompatibleTo(*in_arg_ty, from) &&
           PrecisionCompatibleTo(*in_arg_ty, from) &&
           DeviceCompatibleTo(*in_arg_ty, from) &&
           TargetCompatibleTo(*out_arg_ty, to))) {
        VLOG(4) << "picked, opencl found";
        is_found = true;
      } else if (TypeCompatible(*in_arg_ty, from) &&
                 TargetCompatibleTo(*out_arg_ty, to)) {
        VLOG(4) << "picked";
        is_found = true;
      }

      if (is_found) {
        selected_kernels.emplace_back(std::move(kernel));
        // we pick the kernel
        io_copy_inst->AsStmt(
            io_copy_type, std::move(selected_kernels), io_copy_op);
        (*copied_nodes)[in->AsArg().name] = io_copy_output_arg;
        break;
      }

      VLOG(4) << "not picked";
    }

    CHECK(is_found) << "Can't find a io_copy  kernel for io_copy op: " << from
                    << ":" << in->AsArg().name << " -> " << to << ":"
                    << inst_node->AsStmt().op_info()->Type();
    // Remove the old link
    RemoveDirectedLink(in, inst_node);

    // Update the original instruction OpDesc.
    // Update its input to the io_copy_output_name
    // Add new link, var -> new_inst, new_inst->newarg, newarg->inst
    DirectedLink(in,
                 io_copy_inst);  // [last kernel]'s output -> [io_copy kernel]
    DirectedLink(
        io_copy_inst,
        io_copy_output_arg);  // [io_copy kernel] -> [io_copy kernel]'s output
    DirectedLink(io_copy_output_arg,
                 inst_node);  // [io_copy kernel]'s output -> [current kernel]

    UpdateInstNode(in, graph, inst_node, io_copy_output_name);
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

void TypeTargetTransformPass::SetValidPlaces(
    const std::vector<Place>& valid_places) {
  CHECK(!valid_places.empty());
  valid_places_ = valid_places;
}

void TypeTargetTransformPass::UpdateInstNode(Node* in,
                                             SSAGraph* graph,
                                             Node* inst_node,
                                             std::string io_copy_output_name) {
  // reset opdesc and update kernel information
  UpdateInputs(
      inst_node->AsStmt().op().get(), in->AsArg().name, io_copy_output_name);
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

REGISTER_MIR_PASS(type_target_cast_pass,
                  paddle::lite::mir::TypeTargetTransformPass)
    .BindTargets({TARGET(kAny)});
