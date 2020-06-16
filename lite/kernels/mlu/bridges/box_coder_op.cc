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

#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

inline cnmlBoxCodeType_t GetBoxCodeType(const std::string& type) {
  if (type == "encode_center_size") {
    return cnmlBoxCodeType_t::Encode;
  }
  return cnmlBoxCodeType_t::Decode;
}

int BoxCoderConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  auto Prior_box_name = op_info->Input("PriorBox").front();
  auto Target_box_name = op_info->Input("TargetBox").front();
  auto Output_box_name = op_info->Output("OutputBox").front();
  std::vector<std::string> input_arg_names = op_info->InputArgumentNames();
  if (std::find(input_arg_names.begin(),
                input_arg_names.end(),
                "PriorBoxVar") == input_arg_names.end()) {
    LOG(FATAL) << "box coder mlu kernel expect PriorBoxVar input" << std::endl;
  }
  auto box_var_name = op_info->Input("PriorBoxVar").front();

  auto* prior_box = scope->FindVar(Prior_box_name)->GetMutable<Tensor>();
  auto* target_box = scope->FindVar(Target_box_name)->GetMutable<Tensor>();
  auto* proposals = scope->FindVar(Output_box_name)->GetMutable<Tensor>();
  auto* box_var = scope->FindVar(box_var_name)->GetMutable<Tensor>();

  auto code_type_str = op_info->GetAttr<std::string>("code_type");
  auto box_normalized = op_info->GetAttr<bool>("box_normalized");
  int axis = -1;
  if (op_info->HasAttr("axis")) {
    axis = op_info->GetAttr<int>("axis");
  } else {
    LOG(FATAL) << "box coder mlu kernel expect axis" << std::endl;
  }

  if (op_info->HasAttr("variance")) {
    LOG(WARNING) << "box coder mlu kernel expect not have variance attr"
                 << std::endl;
    VLOG(6) << "variance: ";
    auto variance_vec = op_info->GetAttr<std::vector<float>>("variance");
    for (size_t i = 0; i < variance_vec.size(); i++) {
      VLOG(6) << variance_vec[i];
    }
  }
  cnmlBoxCodeType_t code_type = GetBoxCodeType(code_type_str);

  int row = -1;
  int len = -1;
  int col = -1;
  if (code_type == cnmlBoxCodeType_t::Encode) {
    // target_box_shape = {row, len};
    // prior_box_shape = {col, len};
    // output_shape = {row, col, len};
    row = target_box->dims()[0];
    len = target_box->dims()[1];
    col = prior_box->dims()[0];
  } else if (code_type == cnmlBoxCodeType_t::Decode) {
    // target_box_shape = {row,col,len};
    // prior_box_shape = {col, len} if axis == 0, or {row, len};
    // output_shape = {row, col, len};
    row = target_box->dims()[0];
    col = target_box->dims()[1];
    len = target_box->dims()[2];
    if (axis == 0) {
      CHECK(prior_box->dims()[0] == col);
    } else {
      CHECK(prior_box->dims()[0] == row);
    }
  }

  bool float32_precision = false;
  if (graph->FPType() == CNML_DATA_FLOAT32) {
    float32_precision = true;
  }

  // =================== DEBUG ======================
  VLOG(6) << "prior_box->dims(): " << prior_box->dims();
  VLOG(6) << "target_box->dims(): " << target_box->dims();
  VLOG(6) << "box_var->dims(): " << box_var->dims();
  VLOG(6) << "proposals->dims(): " << proposals->dims();
  VLOG(6) << "code_type_str: " << code_type_str;
  VLOG(6) << "col: " << col;
  VLOG(6) << "row: " << row;
  VLOG(6) << "len: " << len;
  VLOG(6) << "axis: " << axis;
  VLOG(6) << "box_normalized :" << box_normalized;
  VLOG(6) << "float32_precision: " << float32_precision;
  VLOG(6) << "Prior_box_name: " << Prior_box_name;
  VLOG(6) << "Target_box_name: " << Target_box_name;
  VLOG(6) << "Output_box_name: " << Output_box_name;
  VLOG(6) << "box_var_name: " << box_var_name;

  // =================== DEBUG END ======================
  auto target_box_tensor = graph->GetNode(Target_box_name);
  auto prior_box_tensor = graph->GetNode(Prior_box_name);
  auto box_var_tensor = graph->GetNode(box_var_name);
  auto proposals_tensor = graph->AddNode(Output_box_name,
                                         proposals->dims().Vectorize(),
                                         CNML_TENSOR,
                                         CNML_NCHW,
                                         graph->FPType());
  cnmlPluginBoxCoderOpParam_t param;
  CNML_CALL(
      cnmlCreatePluginBoxCoderOpParam(&param,
                                      row,
                                      col,
                                      len,
                                      axis,
                                      box_normalized,
                                      float32_precision,
                                      code_type,
                                      TargetWrapperMlu::MLUCoreVersion()));
  cnmlBaseOp_t box_coder_op;
  cnmlTensor_t input_tensors[3];
  input_tensors[0] = target_box_tensor->mlu_tensor();
  input_tensors[1] = prior_box_tensor->mlu_tensor();
  input_tensors[2] = box_var_tensor->mlu_tensor();
  cnmlTensor_t output_tensors[1];
  output_tensors[0] = proposals_tensor->mlu_tensor();
  CNML_CALL(cnmlCreatePluginBoxCoderOp(
      &box_coder_op, param, input_tensors, output_tensors));

  // CNML_CALL(cnmlSetOperationComputingLayout(box_coder_op, CNML_NCHW)); //
  // important
  graph->FuseOp(box_coder_op);
  cnmlDestroyPluginBoxCoderOpParam(&param);
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(box_coder,
                         kMLU,
                         paddle::lite::subgraph::mlu::BoxCoderConverter);
