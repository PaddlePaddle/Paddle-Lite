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

#include "lite/core/optimizer/mir/op_transformation_pass.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void OpTransformationPass::CopyAttrFromOpInfo(cpp::OpDesc* op_desc,
                                              OpInfo* op_info,
                                              const std::string& attr_name) {
  auto attr_type = op_info->GetAttrType(attr_name);
  switch (attr_type) {
    case OpDescAPI::AttrType::INT:
      op_desc->SetAttr(attr_name, op_info->GetAttr<int>(attr_name));
      break;
    case OpDescAPI::AttrType::FLOAT:
      op_desc->SetAttr(attr_name, op_info->GetAttr<float>(attr_name));
      break;
    case OpDescAPI::AttrType::BOOLEAN:
      op_desc->SetAttr(attr_name, op_info->GetAttr<bool>(attr_name));
      break;
    case OpDescAPI::AttrType::STRING:
      op_desc->SetAttr(attr_name, op_info->GetAttr<std::string>(attr_name));
      break;
    case OpDescAPI::AttrType::FLOATS: {
      op_desc->SetAttr(attr_name,
                       op_info->GetAttr<std::vector<float>>(attr_name));
    } break;
    case OpDescAPI::AttrType::INTS: {
      op_desc->SetAttr(attr_name,
                       op_info->GetAttr<std::vector<int>>(attr_name));
    } break;
    case OpDescAPI::AttrType::STRINGS: {
      op_desc->SetAttr(attr_name,
                       op_info->GetAttr<std::vector<std::string>>(attr_name));
    } break;
    default:
      LOG(FATAL) << ":Unknow type(" << static_cast<int>(attr_type) << ")";
      break;
  }
}

void OpTransformationPass::CopyAllInputsFromOpInfo(cpp::OpDesc* op_desc,
                                                   OpInfo* op_info) {
  std::vector<std::string> input_names = op_info->input_argnames();
  for (auto& name : input_names) {
    op_desc->SetInput(name, op_info->Input(name));
  }
}

void OpTransformationPass::CopyAllOutputsFromOpInfo(cpp::OpDesc* op_desc,
                                                    OpInfo* op_info) {
  std::vector<std::string> output_names = op_info->output_argnames();
  for (auto& name : output_names) {
    op_desc->SetOutput(name, op_info->Output(name));
  }
}

void OpTransformationPass::CopyInputScaleFromOpInfo(cpp::OpDesc* op_desc,
                                                    OpInfo* op_info,
                                                    const std::string& name) {
  if (op_info->HasInputScale(name, true)) {
    op_desc->SetAttr<std::vector<float>>(name,
                                         op_info->GetInputScale(name, true));
  }
}

void OpTransformationPass::CopyOutputScaleFromOpInfo(cpp::OpDesc* op_desc,
                                                     OpInfo* op_info,
                                                     const std::string& name) {
  if (op_info->HasOutputScale(name, true)) {
    op_desc->SetAttr<std::vector<float>>(name,
                                         op_info->GetOutputScale(name, true));
  }
}

void OpTransformationPass::UpdateNodeFromOpdesc(mir::Node* node,
                                                cpp::OpDesc* op_desc) {
  auto new_op = LiteOpRegistry::Global().Create(op_desc->Type());
  new_op->Attach(*op_desc, node->stmt()->op()->scope());
  new_op->SetValidPlaces(node->stmt()->op()->valid_places());
  auto kernels = new_op->CreateKernels(new_op->valid_places());
  node->stmt()->SetOp(new_op);
  node->stmt()->SetKernels(std::move(kernels));
}
void OpTransformationPass::ConvertDepthewiseConv2dTranspose2Conv2dTranspose(
    mir::Node* node) {
  auto* op_info = node->stmt()->mutable_op_info();
  cpp::OpDesc op_desc;
  op_desc.SetType("conv2d_transpose");

  // Copies inputs/outputs/attributes
  CopyAllInputsFromOpInfo(&op_desc, op_info);
  CopyAllOutputsFromOpInfo(&op_desc, op_info);
  std::vector<std::string> attr_names = op_info->AttrNames();
  for (size_t i = 0; i < attr_names.size(); i++) {
    if (op_info->HasAttr(attr_names[i])) {
      CopyAttrFromOpInfo(&op_desc, op_info, attr_names[i]);
    }
  }
  // Copy inputs/outputs scales
  if (op_info->HasAttr("enable_int8")) {
    CopyInputScaleFromOpInfo(&op_desc, op_info, "Input0_scale");
    CopyInputScaleFromOpInfo(&op_desc, op_info, "Filter0_scale");
    CopyOutputScaleFromOpInfo(&op_desc, op_info, "Output0_scale");
  }
  // Update node from op_desc
  UpdateNodeFromOpdesc(node, &op_desc);
}
/*
* Op transformation: We convert some ops into other types to reduce the topology
* complexity
*    transformation 1 :  depthwise_conv2d_transpose  -----> conv2d_transpose
*/
void OpTransformationPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->IsStmt() &&
        node->AsStmt().op_type() == "depthwise_conv2d_transpose") {
      ConvertDepthewiseConv2dTranspose2Conv2dTranspose(node);
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(op_transformation_pass,
                  paddle::lite::mir::OpTransformationPass)
    .BindTargets({TARGET(kARM)});
