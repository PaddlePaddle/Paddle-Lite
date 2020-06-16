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

#include <algorithm>

#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/mlu/bridges/multiclass_nms_api.h"
#include "lite/kernels/mlu/bridges/utility.h"
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/operators/multiclass_nms_op.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

int MulticlassNmsConverter(void* ctx, OpLite* op, KernelBase* kernel) {
  CHECK(ctx != nullptr);
  CHECK(op != nullptr);
  auto graph = static_cast<Graph*>(ctx);
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto scope = op->scope();
  VLOG(3) << "[MLU] Converting " + op_type + "...";

  auto bboxes_name = op_info->Input("BBoxes").front();
  auto scores_name = op_info->Input("Scores").front();
  auto out_name = op_info->Output("Out").front();

  auto* bboxes = scope->FindTensor(bboxes_name);
  auto* scores = scope->FindTensor(scores_name);
  auto* out = scope->FindTensor(out_name);
  auto background_label = op_info->GetAttr<int>("background_label");
  auto keep_top_k = op_info->GetAttr<int>("keep_top_k");
  auto nms_top_k = op_info->GetAttr<int>("nms_top_k");
  auto score_threshold = op_info->GetAttr<float>("score_threshold");
  auto nms_threshold = op_info->GetAttr<float>("nms_threshold");
  auto nms_eta = op_info->GetAttr<float>("nms_eta");
  bool normalized = false;
  if (op_info->HasAttr("normalized")) {
    normalized = op_info->GetAttr<bool>("normalized");
  }

  auto bboxes_dims = bboxes->dims();
  auto scores_dims = scores->dims();

  auto batch_size = bboxes->dims()[0];
  auto num_boxes = bboxes->dims()[1];
  auto class_num = scores->dims()[1];
  keep_top_k = keep_top_k == -1 ? num_boxes : keep_top_k;

  // ?????????????
  int box_size = 4;
  std::vector<int64_t> outs_shape = {batch_size, keep_top_k, box_size + 2};
  const_cast<Tensor*>(out)->Resize(outs_shape);
  auto out_dims = out->dims();

  // LOG(WARNING) << "CORE NUM SHOULD BE 4!!!!" << std::endl;

  int core_num = TargetWrapperMlu::MLUCoreNumber();

  // expect {batch_size, num_boxes, box_size} in compute
  // while {batch_size, box_size,num_boxes} on mlu
  // while {batch_size, num_boxes, box_size} on cpu
  // so mlu  data_flow and mlu compute layout mismatch, should set bboxes_tensor
  // as NCHW
  auto bboxes_tensor = graph->GetNode(bboxes_name);
  // expect {batch_size, class_num, num_boxes} in compute
  // while  {batch_size,  num_boxes,class_num } on mlu
  // while  {batch_size, class_num, num_boxes} on cpu
  // so mlu  data_flow and mlu compute layout mismatch, should set scores_tensor
  // as NCHW
  auto scores_tensor = graph->GetNode(scores_name);
  // expect batch_size, keep_top_k, box_size + 2 in compute
  // while batch_size, box_size + 2, keep_top_k on mlu
  // while batch_size, keep_top_k, box_size + 2 on cpu
  // so mlu  data_flow and mlu compute layout mismatch, should set out_tensor as
  auto out_tensor = graph->AddNode(
      out_name, out_dims.Vectorize(), CNML_TENSOR, CNML_NCHW, graph->FPType());

  // trans bboxes {batch_size, num_boxes, box_size}
  auto bboxes_trans_tensor = graph->AddNode(bboxes_name + ".trans.bboxes",
                                            bboxes_dims.Vectorize(),
                                            CNML_TENSOR,
                                            CNML_NCHW,
                                            graph->FPType(),
                                            CNML_NCHW);
  // trans scores {batch_size, class_num, num_boxes}
  auto scores_trans_tensor = graph->AddNode(bboxes_name + ".trans.scores",
                                            scores_dims.Vectorize(),
                                            CNML_TENSOR,
                                            CNML_NCHW,
                                            graph->FPType(),
                                            CNML_NCHW);
  // trans out {batch_size, keep_top_k, box_size + 2}
  auto out_trans_tensor = graph->AddNode(out_name + ".trans.out",
                                         out_dims.Vectorize(),
                                         CNML_TENSOR,
                                         CNML_NCHW,
                                         graph->FPType(),
                                         CNML_NCHW);

  std::string out_num_name = "nms_out_num";
  auto* out_num = scope->NewTensor(out_num_name);
  std::vector<int64_t> out_num_shape = {batch_size, 1};
  out_num->Resize(out_num_shape);
  auto num_outs_tensor = graph->AddNode(
      out_num_name, out_num_shape, CNML_TENSOR, CNML_NCHW, graph->FPType());
  bool float_precision = false;
  if (graph->FPType() == CNML_DATA_FLOAT32) {
    float_precision = true;
  }
  int64_t workspace_mem_size =
      4 * std::min(static_cast<int>(batch_size), core_num) *
      (14 * num_boxes + 8 * class_num * num_boxes);
  int64_t workspace_fp_size = workspace_mem_size / 4;
  if (!float_precision) {
    // when run as fp16, mlu size will be half of cpu size, so workspace_fp_size
    // should be double
    workspace_fp_size = workspace_mem_size / 2;
  }
  std::vector<int64_t> workspace_shape = {workspace_fp_size};
  std::string nms_workspace_name =
      "nms_workspace";  // expect only one nms in same model
  auto workspace_tensor = graph->AddNode(nms_workspace_name,
                                         workspace_shape,
                                         CNML_CONST,
                                         CNML_NCHW,
                                         graph->FPType());
  std::vector<float> workspace_cpu(workspace_shape[0]);
  // void* work_space_ = nullptr;
  // cnrtMalloc(&work_space_, workspace_shape[0]);
  VLOG(6) << "workspace_shape :" << workspace_shape[0];
  // VLOG(6) << "workspace_shape mlu ptr :"
  //         << reinterpret_cast<void*>(work_space_);

  // =================== Bboxes Trans ============================
  std::vector<int> bboxes_axis = {0, 2, 1};
  cnmlBaseOp_t bboxes_trans_op{nullptr};
  cnmlNdTransposeOpParam_t bboxes_trans_param{nullptr};
  CNML_CALL(cnmlCreateNdTransposeOpParam(
      &bboxes_trans_param, bboxes_axis.data(), bboxes_axis.size()));
  CNML_CALL(cnmlCreateNdTransposeProOp(&bboxes_trans_op,
                                       bboxes_tensor->mlu_tensor(),
                                       bboxes_trans_tensor->mlu_tensor(),
                                       bboxes_trans_param));
  // =================== Bboxes Trans END ========================

  // =================== Scores Trans ============================
  std::vector<int> scores_axis = {0, 2, 1};
  cnmlBaseOp_t scores_trans_op{nullptr};
  cnmlNdTransposeOpParam_t scores_trans_param{nullptr};
  CNML_CALL(cnmlCreateNdTransposeOpParam(
      &scores_trans_param, scores_axis.data(), scores_axis.size()));
  CNML_CALL(cnmlCreateNdTransposeProOp(&scores_trans_op,
                                       scores_tensor->mlu_tensor(),
                                       scores_trans_tensor->mlu_tensor(),
                                       scores_trans_param));
  // =================== Scores Trans END ========================
  multiclass_nms_param_t params_;
  create_multiclass_nms_param(&params_,
                              score_threshold,
                              nms_top_k,
                              keep_top_k,
                              nms_threshold,
                              normalized,
                              nms_eta,
                              background_label,
                              batch_size,
                              class_num,
                              num_boxes,
                              box_size);

  cnmlBaseOp_t multiclass_nms_op;
  create_multiclass_nms_op(&multiclass_nms_op,
                           params_,
                           bboxes_trans_tensor->mlu_tensor(),
                           scores_trans_tensor->mlu_tensor(),
                           out_trans_tensor->mlu_tensor(),
                           num_outs_tensor->mlu_tensor(),
                           workspace_tensor->mlu_tensor(),
                           float_precision);

  graph->BindConstRawData(
      nms_workspace_name, workspace_cpu.data(), workspace_cpu.size(), true);

  // =================== Out Trans ============================
  std::vector<int> out_axis = {0, 2, 1};
  cnmlBaseOp_t out_trans_op{nullptr};
  cnmlNdTransposeOpParam_t out_trans_param{nullptr};
  CNML_CALL(cnmlCreateNdTransposeOpParam(
      &out_trans_param, out_axis.data(), out_axis.size()));
  CNML_CALL(cnmlCreateNdTransposeProOp(&out_trans_op,
                                       out_trans_tensor->mlu_tensor(),
                                       out_tensor->mlu_tensor(),
                                       out_trans_param));
  // =================== Out Trans END ========================

  // =================== DEBUG ====================
  VLOG(6) << "bboxes_name: " << bboxes_name;
  VLOG(6) << "scores_name: " << scores_name;
  VLOG(6) << "out_name: " << out_name;
  VLOG(6) << "background_label: " << background_label;
  VLOG(6) << "keep_top_k: " << keep_top_k;
  VLOG(6) << "nms_top_k: " << nms_top_k;
  VLOG(6) << "score_threshold: " << score_threshold;
  VLOG(6) << "nms_threshold: " << nms_threshold;
  VLOG(6) << "nms_eta: " << nms_eta;
  VLOG(6) << "normalized: " << normalized;
  VLOG(6) << "bboxes_dims: " << bboxes_dims;
  VLOG(6) << "scores_dims: " << scores_dims;
  VLOG(6) << "out_dims: " << out_dims;
  VLOG(6) << "out_dims: " << out->dims();
  VLOG(6) << "batch_size: " << batch_size;
  VLOG(6) << "num_boxes : " << num_boxes;
  VLOG(6) << "class_num: " << class_num;
  //   cnmlPrintTensor(bboxes_tensor->mlu_tensor(), CNML_TENSOR);
  //   cnmlPrintTensor(bboxes_trans_tensor->mlu_tensor(), CNML_TENSOR);
  //   cnmlPrintTensor(scores_tensor->mlu_tensor(), CNML_TENSOR);
  //   cnmlPrintTensor(scores_trans_tensor->mlu_tensor(), CNML_TENSOR);
  //   cnmlPrintTensor(out_tensor->mlu_tensor(), CNML_TENSOR);
  //   cnmlPrintTensor(out_trans_tensor->mlu_tensor(), CNML_TENSOR);
  //   cnmlPrintTensor(num_outs_tensor->mlu_tensor(), CNML_TENSOR);
  // =================== DEBUG END ================
  graph->FuseOp(bboxes_trans_op);
  graph->FuseOp(scores_trans_op);
  graph->FuseOp(multiclass_nms_op);
  graph->FuseOp(out_trans_op);
  return SUCCESS;
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(multiclass_nms,
                         kMLU,
                         paddle::lite::subgraph::mlu::MulticlassNmsConverter);
