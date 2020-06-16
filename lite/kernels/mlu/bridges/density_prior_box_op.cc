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

void inferShape(Tensor* input,
                Tensor* boxes,
                Tensor* variances,
                std::vector<float> fixed_ratios,
                std::vector<int> densities) {
  auto feat_height = input->dims()[2];
  auto feat_width = input->dims()[3];

  int num_priors = 0;
  for (size_t i = 0; i < densities.size(); ++i) {
    num_priors += (fixed_ratios.size()) * (pow(densities[i], 2));
  }

  std::vector<int64_t> boxes_shape = {feat_width, feat_height, num_priors, 4};
  std::vector<int64_t> vars_shape = boxes_shape;
  boxes->Resize(boxes_shape);
  variances->Resize(vars_shape);
}

int DensityPriorBoxConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  auto input_name = op_info->Input("Input").front();
  auto image_name = op_info->Input("Image").front();
  auto boxes_name = op_info->Output("Boxes").front();
  auto variances_name = op_info->Output("Variances").front();

  auto input_var = scope->FindVar(input_name)->GetMutable<Tensor>();
  auto image_var = scope->FindVar(image_name)->GetMutable<Tensor>();
  auto boxes_var = scope->FindVar(boxes_name)->GetMutable<Tensor>();
  auto variances_var = scope->FindVar(variances_name)->GetMutable<Tensor>();

  auto clip = op_info->GetAttr<bool>("clip");
  auto fixed_sizes = op_info->GetAttr<std::vector<float>>("fixed_sizes");
  auto fixed_ratios = op_info->GetAttr<std::vector<float>>("fixed_ratios");
  auto variances_ = op_info->GetAttr<std::vector<float>>("variances");
  auto densities = op_info->GetAttr<std::vector<int>>("densities");
  auto offset = op_info->GetAttr<float>("offset");
  auto step_w = op_info->GetAttr<float>("step_w");
  auto step_h = op_info->GetAttr<float>("step_h");

  inferShape(input_var, boxes_var, variances_var, fixed_ratios, densities);

  auto input_dims = input_var->dims();
  auto image_dims = image_var->dims();
  auto boxes_dims = boxes_var->dims();
  auto variances_dims = variances_var->dims();

  auto feat_tensor = graph->GetNode(input_name);
  auto image_tensor = graph->GetNode(image_name);

  auto boxes_tensor_trans = graph->AddNode(boxes_name + ".trans.boxes",
                                           boxes_dims.Vectorize(),
                                           CNML_TENSOR,
                                           CNML_NHWC,
                                           graph->FPType());
  auto variances_tensor_trans = graph->AddNode(variances_name + ".trans.vars",
                                               variances_dims.Vectorize(),
                                               CNML_TENSOR,
                                               CNML_NHWC,
                                               graph->FPType());

  bool float32_precision = false;
  if (graph->FPType() == CNML_DATA_FLOAT32) {
    float32_precision = true;
  }

  // ==================== DEBUG ==================

  VLOG(6) << "input_name: " << input_name;
  VLOG(6) << "image_name: " << image_name;
  VLOG(6) << "boxes_name: " << boxes_name;
  VLOG(6) << "variances_name: " << variances_name;
  VLOG(6) << "input_dims : " << input_dims;
  VLOG(6) << "image_dims : " << image_dims;
  VLOG(6) << "boxes_dims : " << boxes_dims;
  VLOG(6) << "variances_dims : " << variances_dims;
  VLOG(6) << "clip : " << clip;
  VLOG(6) << "fixed_sizes : ";
  for (auto tmp : fixed_sizes) {
    VLOG(6) << tmp;
  }

  VLOG(6) << "fixed_ratios : ";
  for (auto tmp : fixed_ratios) {
    VLOG(6) << tmp;
  }
  VLOG(6) << "variances_ : ";
  for (auto tmp : variances_) {
    VLOG(6) << tmp;
  }
  VLOG(6) << "densities : ";
  for (auto tmp : densities) {
    VLOG(6) << tmp;
  }
  VLOG(6) << "offset : " << offset;
  VLOG(6) << "clip : " << clip;

  int cnml_boxes_shape[4];
  CNML_CALL(
      cnmlGetTensorShape(boxes_tensor_trans->mlu_tensor(), cnml_boxes_shape));
  VLOG(6) << "cnml_boxes_shape";
  for (size_t i = 0; i < 4; i++) {
    VLOG(6) << cnml_boxes_shape[i];
  }
  int cnml_vars_shape[4];
  VLOG(6) << "cnml_vars_shape";
  CNML_CALL(cnmlGetTensorShape(variances_tensor_trans->mlu_tensor(),
                               cnml_vars_shape));
  for (size_t i = 0; i < 4; i++) {
    VLOG(6) << cnml_vars_shape[i];
  }

  int feat_width = input_dims[3];
  int feat_height = input_dims[2];
  int image_width = image_dims[3];
  int image_height = image_dims[2];
  // ==================== DEBUG END ==================
  cnmlPluginDensityPriorBoxOpParam_t op_param;
  cnmlCreatePluginDensityPriorBoxOpParam(&op_param,
                                         feat_width,
                                         feat_height,
                                         image_width,
                                         image_height,
                                         variances_.data(),
                                         variances_.size(),
                                         densities.data(),
                                         densities.size(),
                                         fixed_sizes.data(),
                                         fixed_sizes.size(),
                                         fixed_ratios.data(),
                                         fixed_ratios.size(),
                                         clip,
                                         step_w,
                                         step_h,
                                         offset,
                                         float32_precision,
                                         TargetWrapperMlu::MLUCoreVersion());

  cnmlTensor_t input_tensors[2];
  input_tensors[0] = feat_tensor->mlu_tensor();
  input_tensors[1] = image_tensor->mlu_tensor();
  cnmlTensor_t output_tensors[2];
  output_tensors[0] = boxes_tensor_trans->mlu_tensor();
  output_tensors[1] = variances_tensor_trans->mlu_tensor();
  cnmlBaseOp_t density_prior_box_op;
  CNML_CALL(cnmlCreatePluginDensityPriorBoxOp(
      &density_prior_box_op, op_param, input_tensors, output_tensors));

  std::vector<int> nchw_to_nhwc_axis = {0, 2, 3, 1};
  // ============== Boxes Trans =======================
  auto boxes_tensor = graph->AddNode(boxes_name,
                                     boxes_dims.Vectorize(),
                                     CNML_TENSOR,
                                     CNML_NCHW,
                                     graph->FPType());
  cnmlBaseOp_t trans_boxes_op{nullptr};
  cnmlNdTransposeOpParam_t trans_boxes_param{nullptr};
  CNML_CALL(cnmlCreateNdTransposeOpParam(
      &trans_boxes_param, nchw_to_nhwc_axis.data(), nchw_to_nhwc_axis.size()));
  CNML_CALL(cnmlCreateNdTransposeProOp(&trans_boxes_op,
                                       boxes_tensor_trans->mlu_tensor(),
                                       boxes_tensor->mlu_tensor(),
                                       trans_boxes_param));
  // ============== Boxes Trans End ===================

  // ============== Vars Trans =======================
  auto variances_tensor = graph->AddNode(variances_name,
                                         variances_dims.Vectorize(),
                                         CNML_TENSOR,
                                         CNML_NCHW,
                                         graph->FPType());
  cnmlBaseOp_t trans_vars_op{nullptr};
  cnmlNdTransposeOpParam_t trans_vars_param{nullptr};
  CNML_CALL(cnmlCreateNdTransposeOpParam(
      &trans_vars_param, nchw_to_nhwc_axis.data(), nchw_to_nhwc_axis.size()));
  CNML_CALL(cnmlCreateNdTransposeProOp(&trans_vars_op,
                                       variances_tensor_trans->mlu_tensor(),
                                       variances_tensor->mlu_tensor(),
                                       trans_vars_param));
  // ============== Vars Trans End ===================

  // cnmlSetOperationComputingLayout(density_prior_box_op,CNML_NCHW);
  // cnmlSetTensorComputingLayoutInOperation(
  //     density_prior_box_op, boxes_tensor->mlu_tensor(), CNML_NCHW);
  // cnmlSetTensorComputingLayoutInOperation(
  //     density_prior_box_op, variances_tensor->mlu_tensor(), CNML_NCHW);
  graph->FuseOp(trans_boxes_op);
  graph->FuseOp(density_prior_box_op);
  graph->FuseOp(trans_vars_op);
  // cnmlDestroyPluginDensityPriorBoxOpParam(&op_param);
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(density_prior_box,
                         kMLU,
                         paddle::lite::subgraph::mlu::DensityPriorBoxConverter);
