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
#include "lite/operators/subgraph_op.h"

namespace paddle {
namespace lite {
namespace mir {

// For the subgraph op, we also need to update the attr 'input_data_names' and
// the input variables names of the Ops in the subblock.
void UpdateInputsForSubgraph(OpLite* op,
                             const std::string& from,
                             const std::string& to) {
  auto* op_desc = op->mutable_op_info();
  auto input_data_names =
      op_desc->GetAttr<std::vector<std::string>>("input_data_names");
  std::replace(input_data_names.begin(), input_data_names.end(), from, to);
  op_desc->SetAttr("input_data_names", input_data_names);
  auto sub_program_desc =
      static_cast<operators::SubgraphOp*>(op)->GetProgramDesc();
  CHECK(sub_program_desc);
  int sub_block_idx = op_desc->GetAttr<int32_t>("sub_block");
  auto sub_block_desc =
      sub_program_desc->GetBlock<cpp::BlockDesc>(sub_block_idx);
  for (size_t sub_op_idx = 0; sub_op_idx < sub_block_desc->OpsSize();
       sub_op_idx++) {
    auto sub_op_desc = const_cast<cpp::OpDesc*>(
        sub_block_desc->GetOp<cpp::OpDesc>(sub_op_idx));
    for (auto& sub_op_input : *sub_op_desc->mutable_inputs()) {
      for (auto& sub_var_name : sub_op_input.second) {
        if (sub_var_name == from) {
          sub_var_name = to;
        }
      }
    }
  }
}

// Update the input variable names from 'from' to 'to' for the target Op
void UpdateInputs(OpLite* op, const std::string& from, const std::string& to) {
  auto* op_desc = op->mutable_op_info();
  auto op_type = op_desc->Type();
  for (auto& op_input : *op_desc->mutable_inputs()) {
    for (auto& var_name : op_input.second) {
      if (var_name == from) {
        var_name = to;
      }
    }
  }
  if (op_type == "subgraph") {
    UpdateInputsForSubgraph(op, from, to);
  }
}

// Infer the scale value for the new calib op from the subgraph op
static bool InferScaleFromSubgraph(std::string var_name,
                                   const OpInfo* op_info,
                                   float* scale,
                                   bool reverse = false) {
  std::string attr_name = reverse ? "output_data_names" : "input_data_names";
  if (!op_info->HasAttr(attr_name)) return false;
  auto input_or_output_names =
      op_info->GetAttr<std::vector<std::string>>(attr_name);
  attr_name = reverse ? "output_data_scales" : "input_data_scales";
  if (!op_info->HasAttr(attr_name)) return false;
  auto input_or_output_scales = op_info->GetAttr<std::vector<float>>(attr_name);
  auto size = input_or_output_names.size();
  CHECK(size == input_or_output_scales.size());
  for (size_t i = 0; i < size; i++) {
    if (input_or_output_names[i] == var_name) {
      *scale = input_or_output_scales[i];
      return true;
    }
  }
  return false;
}

// Infer the scale value for the new calib op from the input_scale of the
// current op and output_scale of the previous op.
// case 1: prev_op->var_node->op_node(int8->any op, with input_scale).
// case 2: prev_op->var_node->op_node(subgraph op, int8->any, with
// input_data_scales).
// case 3: prev_op(any->int8, with output_scale)->var_node->op_node(fp32->any,
// without input_scale).
// case 4: prev_op(any->int8, subgraph_op, with
// output_data_scales)->var_node->op_node(fp32->any, without input_scale).
static bool InferScale(Node* var_node, Node* op_node, float* scale) {
  bool found = false;
  auto& inst = op_node->AsStmt();
  auto op_info = inst.op_info();
  auto op_type = op_info->Type();
  auto var_name = var_node->AsArg().name;
  if (op_type == "subgraph") {
    found = InferScaleFromSubgraph(var_name, op_info, scale, false);
  } else {
    if (op_info->HasAttr("input_scale")) {
      *scale = op_info->GetAttr<float>("input_scale");
      found = true;
    } else {
      // Obtain the output_scale from one of its previous Ops
      auto prev_op_node = var_node->inlinks.front();
      CHECK(prev_op_node->IsStmt());
      auto& prev_inst = prev_op_node->AsStmt();
      auto prev_op_info = prev_inst.op_info();
      auto prev_op_type = prev_op_info->Type();
      if (prev_op_type == "subgraph") {
        found = InferScaleFromSubgraph(var_name, prev_op_info, scale, true);
      } else {
        if (prev_op_info->HasAttr("output_scale")) {
          *scale = prev_op_info->GetAttr<float>("output_scale");
          found = true;
        }
      }
    }
  }
  return found;
}

void PrecisionCastPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  // Start from inputs of the graph, those should have place set.
  std::list<Node*> nodes;
  for (auto& node : graph->StmtTopologicalOrder()) {
    nodes.push_back(node);
  }

  // record the copied node.
  std::unordered_map<std::string, Node*> cast_nodes;

  for (auto& node : nodes) {
    if (!node->IsStmt() || node->AsStmt().op_type() == "while") continue;
    auto inlinks = node->inlinks;
    for (auto* in : inlinks) {
      ComplementInputs(graph.get(), node, in, &cast_nodes);
    }
  }
}

void PrecisionCastPass::ComplementInputs(
    SSAGraph* graph,
    Node* inst_node,
    Node* in,
    std::unordered_map<std::string, Node*>* cast_nodes) {
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
  if (inst.op_info()->Type() == "fetch") {
    if (inst.op_info()->HasAttr("data_type")) {
      auto data_type =
          static_cast<PrecisionType>(inst.op_info()->GetAttr<int>("data_type"));
      decl_arg_type = LiteType::GetTensorTy(
          decl_arg_type->target(), data_type, decl_arg_type->layout());
    }
  }
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
                cast_nodes,
                graph->valid_places());
  }
}

void PrecisionCastPass::AddCastInst(
    const Type& from,
    const Type& to,
    Node* in,
    SSAGraph* graph,
    Node* inst_node,
    std::unordered_map<std::string, Node*>* cast_nodes,
    const std::vector<Place>& valid_places) {
  CHECK(!valid_places.empty()) << "valid_place should be set";

  // var -> new_transform_op -> new_var -> inst
  // So there will be a new Argument node and a new Cast Statement Node.
  CHECK(in->IsArg());
  // auto node_id = [&] { return graph->nodes().size(); };
  auto cast_op_output_name = in->AsArg().name + "/precision_trans";
  // in->AsArg().name + "/precision_trans/" +
  // paddle::lite::to_string(node_id());
  if (cast_nodes->count(in->AsArg().name)) {
    // Remove the old link
    RemoveDirectedLink(in, inst_node);
    // Update the original instruction OpDesc.
    // Update its input to the cast_op_output_name
    // Add new link, newarg->inst
    DirectedLink(cast_nodes->at(in->AsArg().name),
                 inst_node);  // [io_copy kernel]'s output -> [current kernel]
    // reset opdesc and update kernel information
    UpdateInputs(
        inst_node->AsStmt().op().get(), in->AsArg().name, cast_op_output_name);
  } else {
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
    float scale;
    if (InferScale(in, inst_node, &scale)) {
      op_desc.SetAttr("scale", scale);
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
        (*cast_nodes)[in->AsArg().name] = cast_op_output_arg;
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
    UpdateInputs(
        inst_node->AsStmt().op().get(), in->AsArg().name, cast_op_output_name);
  }

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
    .ExcludeTargets({TARGET(kOpenCL)})
    .BindKernel("calib_once")
    .BindKernel("calib");
