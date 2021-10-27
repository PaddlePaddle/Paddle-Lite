// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/optimizer/mir/nnadapter_insert_calib_pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

static bool is_quant_node(Node* node) {
  CHECK(node->IsStmt());
  auto op_info = node->AsStmt().op_info();
  return op_info->HasAttr("enable_int8") &&
         op_info->GetAttr<bool>("enable_int8");
}

// Update the input variable names from 'from' to 'to' for the target Op
static void UpdateInputs(const std::shared_ptr<paddle::lite::OpLite>& op,
                         const std::string& from,
                         const std::string& to) {
  auto* op_desc = op->mutable_op_info();
  auto op_type = op_desc->Type();
  for (auto& op_input : *op_desc->mutable_inputs()) {
    for (auto& var_name : op_input.second) {
      if (var_name == from) {
        var_name = to;
      }
    }
  }
}

void NNAdapterInsertCalibPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
#if defined(LITE_ON_MODEL_OPTIMIZE_TOOL) || defined(LITE_WITH_PYTHON) || \
    defined(LITE_WITH_NNADAPTER)
  Scope* scope = nullptr;
  for (auto& node : graph->mutable_nodes()) {
    if (node.IsStmt()) {
      scope = node.AsStmt().op()->scope();
      break;
    }
  }
  CHECK(scope);
  auto device_names =
      Context<TargetType::kNNAdapter>::NNAdapterDeviceNames(scope);
  if (std::find(device_names.begin(),
                device_names.end(),
                "huawei_ascend_npu") != device_names.end()) {
    VLOG(3) << "not insert calib nodes.";
    return;
  }
#endif

  auto nodes = graph->StmtTopologicalOrder();
  // Record casted var node to avoid inserting same calib node
  std::map<std::string, Node*> cast_nodes;
  std::vector<std::string> skip_ops = {"while", "conditional_block"};

  for (auto& node : nodes) {
    auto op_type = node->AsStmt().op_type();
    auto iter = std::find(skip_ops.begin(), skip_ops.end(), op_type);
    if (!node->IsStmt() || iter != skip_ops.end()) continue;
    auto inlinks = node->inlinks;
    for (auto* in : inlinks) {
      ComplementInputs(graph, in, node, &cast_nodes);
    }
  }
}

void NNAdapterInsertCalibPass::ComplementInputs(
    const std::unique_ptr<SSAGraph>& graph,
    Node* in,
    Node* inst_node,
    std::map<std::string, Node*>* cast_nodes) {
  CHECK(inst_node->IsStmt());
  CHECK(in->IsArg());
  if (in->inlinks.empty()) return;
  auto pre_node = in->inlinks.front();
  if (is_quant_node(pre_node) == is_quant_node(inst_node)) return;

  if (is_quant_node(pre_node)) {
    in->AsArg().type = LiteType::GetTensorTy(
        TARGET(kNNAdapter), PRECISION(kInt8), DATALAYOUT(kNCHW));
  }
  std::string calib_in_name = in->AsArg().name;
  std::string calib_out_name = calib_in_name + "/nnadapter_precision_trans";
  if (cast_nodes->count(calib_out_name) > 0) {
    RemoveDirectedLink(in, inst_node);
    DirectedLink(cast_nodes->at(calib_out_name), inst_node);
    UpdateInputs(inst_node->AsStmt().op(), calib_in_name, calib_out_name);
  } else {
    // Creat calib out node
    auto calib_out_arg = graph->NewArgumentNode(calib_out_name);
    if (is_quant_node(inst_node)) {
      calib_out_arg->AsArg().type = LiteType::GetTensorTy(
          TARGET(kNNAdapter), PRECISION(kInt8), DATALAYOUT(kNCHW));
    } else {
      calib_out_arg->AsArg().type = LiteType::GetTensorTy(
          TARGET(kNNAdapter), PRECISION(kFloat), DATALAYOUT(kNCHW));
    }
    (*cast_nodes)[calib_out_name] = calib_out_arg;

    // Create calib node
    auto calib_inst = graph->NewInstructNode();
    std::string calib_type{"calib"};
    auto calib_op = LiteOpRegistry::Global().Create(calib_type);
    CHECK(calib_op) << "create op [" << calib_op << "] failed";
    cpp::OpDesc op_desc;
    op_desc.SetType(calib_type);
    op_desc.SetInput("Input", {calib_in_name});
    op_desc.SetOutput("Out", {calib_out_name});
    auto pre_op_info = pre_node->AsStmt().op_info();
    CHECK(pre_op_info->HasOutputScale(calib_in_name));
    auto scales = pre_op_info->GetOutputScale(calib_in_name);
    CHECK(!scales.empty());
    op_desc.SetAttr("scale", scales[0]);
    calib_op->Attach(op_desc, inst_node->AsStmt().op()->scope());
    calib_inst->AsStmt(calib_type, {}, calib_op);

    // in -> calib_inst -> calib_out_arg -> inst_node
    RemoveDirectedLink(in, inst_node);
    DirectedLink(in, calib_inst);
    DirectedLink(calib_inst, calib_out_arg);
    DirectedLink(calib_out_arg, inst_node);
    UpdateInputs(inst_node->AsStmt().op(), calib_in_name, calib_out_name);
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(nnadapter_insert_calib_pass,
                  paddle::lite::mir::NNAdapterInsertCalibPass)
    .BindTargets({TARGET(kNNAdapter)});
