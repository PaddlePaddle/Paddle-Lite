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

static bool IsQuantInstNode(Node* node) {
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
                "huawei_ascend_npu") == device_names.end()) {
    // Only huawei_ascend_npu to test pass
    VLOG(3) << "not insert calib nodes.";
    return;
  }
#endif

  UpdateQuantOpOut(graph);
  InsetQuantCalib(graph);
  InsetDequantCalib(graph);
}

void NNAdapterInsertCalibPass::UpdateQuantOpOut(
    const std::unique_ptr<SSAGraph>& graph) {
  auto nodes = graph->StmtTopologicalOrder();
  for (auto node : nodes) {
    if (!IsQuantInstNode(node)) continue;
    for (auto out_node : node->outlinks) {
      auto& out_type = out_node->AsArg().type;
      // TODO(zhupengyang): Only support trans to int8. Uint8 should be
      // considered.
      out_type = LiteType::GetTensorTy(
          out_type->target(), PRECISION(kInt8), out_type->layout());
    }
  }
}

void NNAdapterInsertCalibPass::InsetQuantCalib(
    const std::unique_ptr<SSAGraph>& graph) {
  auto nodes = graph->StmtTopologicalOrder();
  // Record arg nodes to reuse if other inst nodes need the same arg node
  std::map<std::string, Node*> transed_arg_nodes;
  // Skip if pre op is feed, calib, ...
  std::vector<std::string> skip_pre_ops{"feed", "calib"};

  for (auto node : nodes) {
    if (!IsQuantInstNode(node)) continue;
    for (auto pre_arg_node : node->inlinks) {
      if (pre_arg_node->inlinks.empty()) continue;
      auto pre_inst_node = pre_arg_node->inlinks.front();
      if (IsQuantInstNode(pre_inst_node)) continue;
      if (std::find(skip_pre_ops.begin(),
                    skip_pre_ops.end(),
                    pre_inst_node->AsStmt().op_type()) != skip_pre_ops.end()) {
        continue;
      }

      std::string calib_in_name = pre_arg_node->AsArg().name;
      std::string calib_out_name = calib_in_name + "/quant";
      if (transed_arg_nodes.count(calib_in_name) > 0) {
        // Create topology
        RemoveDirectedLink(pre_arg_node, node);
        DirectedLink(transed_arg_nodes[calib_in_name], node);
        UpdateInputs(node->AsStmt().op(), calib_in_name, calib_out_name);
      } else {
        // Creat calib out node
        auto calib_out_arg = graph->NewArgumentNode(calib_out_name);
        calib_out_arg->AsArg().type = LiteType::GetTensorTy(
            TARGET(kHost), PRECISION(kInt8), DATALAYOUT(kNCHW));
        // Create calib node
        auto calib_inst = graph->NewInstructNode();
        std::string calib_type{"calib"};
        auto calib_op = LiteOpRegistry::Global().Create(calib_type);
        CHECK(calib_op) << "create op [" << calib_op << "] failed";
        cpp::OpDesc op_desc;
        op_desc.SetType(calib_type);
        op_desc.SetInput("Input", {calib_in_name});
        op_desc.SetOutput("Out", {calib_out_name});
        auto op_info = node->AsStmt().op_info();
        CHECK(op_info->HasInputScale(calib_in_name));
        auto scales = op_info->GetInputScale(calib_in_name);
        CHECK_EQ(scales.size(), 1UL);
        op_desc.SetAttr("scale", scales[0]);
        calib_op->Attach(op_desc, node->AsStmt().op()->scope());
        calib_inst->AsStmt(calib_type, {}, calib_op);
        // Create topology
        RemoveDirectedLink(pre_arg_node, node);
        DirectedLink(pre_arg_node, calib_inst);
        DirectedLink(calib_inst, calib_out_arg);
        DirectedLink(calib_out_arg, node);
        UpdateInputs(node->AsStmt().op(), calib_in_name, calib_out_name);
        transed_arg_nodes[calib_in_name] = calib_out_arg;
      }
    }
  }
}

void NNAdapterInsertCalibPass::InsetDequantCalib(
    const std::unique_ptr<SSAGraph>& graph) {
  auto nodes = graph->StmtTopologicalOrder();
  // Record arg nodes to reuse if other inst nodes need the same arg node
  std::map<std::string, Node*> transed_arg_nodes;
  // Skip if op is fetch, calib, ...
  std::vector<std::string> skip_ops{"fetch", "calib"};

  for (auto node : nodes) {
    if (IsQuantInstNode(node)) continue;
    if (std::find(skip_ops.begin(), skip_ops.end(), node->AsStmt().op_type()) !=
        skip_ops.end()) {
      continue;
    }
    for (auto pre_arg_node : node->inlinks) {
      if (pre_arg_node->inlinks.empty()) continue;
      auto pre_inst_node = pre_arg_node->inlinks.front();
      if (!IsQuantInstNode(pre_inst_node)) continue;

      std::string calib_in_name = pre_arg_node->AsArg().name;
      std::string calib_out_name = calib_in_name + "/dequant";
      if (transed_arg_nodes.count(calib_in_name) > 0) {
        // Create topology
        RemoveDirectedLink(pre_arg_node, node);
        DirectedLink(transed_arg_nodes[calib_in_name], node);
        UpdateInputs(node->AsStmt().op(), calib_in_name, calib_out_name);
      } else {
        // Creat calib out node
        auto calib_out_arg = graph->NewArgumentNode(calib_out_name);
        calib_out_arg->AsArg().type = LiteType::GetTensorTy(
            TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
        // Create calib node
        auto calib_inst = graph->NewInstructNode();
        std::string calib_type{"calib"};
        auto calib_op = LiteOpRegistry::Global().Create(calib_type);
        CHECK(calib_op) << "create op [" << calib_op << "] failed";
        cpp::OpDesc op_desc;
        op_desc.SetType(calib_type);
        op_desc.SetInput("Input", {calib_in_name});
        op_desc.SetOutput("Out", {calib_out_name});
        auto pre_op_info = pre_inst_node->AsStmt().op_info();
        CHECK(pre_op_info->HasOutputScale(calib_in_name));
        auto scales = pre_op_info->GetOutputScale(calib_in_name);
        CHECK_EQ(scales.size(), 1UL);
        op_desc.SetAttr("scale", scales[0]);
        calib_op->Attach(op_desc, node->AsStmt().op()->scope());
        calib_inst->AsStmt(calib_type, {}, calib_op);
        // Create topology
        RemoveDirectedLink(pre_arg_node, node);
        DirectedLink(pre_arg_node, calib_inst);
        DirectedLink(calib_inst, calib_out_arg);
        DirectedLink(calib_out_arg, node);
        UpdateInputs(node->AsStmt().op(), calib_in_name, calib_out_name);
        transed_arg_nodes[calib_in_name] = calib_out_arg;
      }
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(nnadapter_insert_calib_pass,
                  paddle::lite::mir::NNAdapterInsertCalibPass)
    .BindTargets({TARGET(kNNAdapter)});
