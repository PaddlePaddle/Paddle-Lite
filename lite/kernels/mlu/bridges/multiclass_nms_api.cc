// Copyright (c) 2020 smarsu. All Rights Reserved.

#include "lite/kernels/mlu/bridges/multiclass_nms_api.h"
#include <cnml.h>
#include <cnrt.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <vector>

extern "C" {
void multiclass_nms_paddle_entry(void *bboxes,
                                 void *scores,
                                 void *outs,
                                 void *num_outs,
                                 float score_threshold,
                                 int nms_top_k,
                                 int keep_top_k,
                                 float nms_threshold,
                                 bool normalized,
                                 float nms_eta,
                                 int background_label,
                                 int batch_size,
                                 int class_num,
                                 int num_boxes,
                                 int box_size,
                                 void *work_space,
                                 DataType data_type);
}  // extern "C"

void create_multiclass_nms_param(multiclass_nms_param_t *params_ptr,
                                 float score_threshold,
                                 int nms_top_k,
                                 int keep_top_k,
                                 float nms_threshold,
                                 bool normalized,
                                 float nms_eta,
                                 int background_label,
                                 int batch_size,
                                 int class_num,
                                 int num_boxes,
                                 int box_size) {
  multiclass_nms_param_t params =
      (multiclass_nms_param_t)malloc(sizeof(struct multiclass_nms_param));
  params->score_threshold = score_threshold;
  params->nms_top_k = nms_top_k;
  params->keep_top_k = keep_top_k;
  params->nms_threshold = nms_threshold;
  params->normalized = normalized;
  params->nms_eta = nms_eta;
  params->background_label = background_label;
  params->batch_size = batch_size;
  params->class_num = class_num;
  params->num_boxes = num_boxes;
  params->box_size = box_size;
  *params_ptr = params;

  return;
}

void destory_multiclass_nms_param(multiclass_nms_param_t *params) {
  if (*params != NULL) {
    free(*params);
  }
}

int create_multiclass_nms_op(cnmlBaseOp_t *op_ptr,
                             multiclass_nms_param_t nms_param,
                             cnmlTensor_t bboxes,
                             cnmlTensor_t scores,
                             cnmlTensor_t outs,
                             cnmlTensor_t num_outs,
                             cnmlTensor_t workspace_tensor,
                             bool float_precision) {
  DataType data_type = kFloat16;
  if (float_precision) {
    data_type = kFloat32;
  }

  if (nms_param->keep_top_k == -1) {
    nms_param->keep_top_k = nms_param->num_boxes;
  }

  cnrtKernelParamsBuffer_t params;
  cnrtGetKernelParamsBuffer(&params);
  cnrtKernelParamsBufferMarkInput(params);
  cnrtKernelParamsBufferMarkInput(params);
  cnrtKernelParamsBufferMarkOutput(params);
  cnrtKernelParamsBufferMarkOutput(params);
  cnrtKernelParamsBufferAddParam(
      params, &nms_param->score_threshold, sizeof(float));
  cnrtKernelParamsBufferAddParam(params, &nms_param->nms_top_k, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &nms_param->keep_top_k, sizeof(int));
  cnrtKernelParamsBufferAddParam(
      params, &nms_param->nms_threshold, sizeof(float));
  cnrtKernelParamsBufferAddParam(params, &nms_param->normalized, sizeof(bool));
  cnrtKernelParamsBufferAddParam(params, &nms_param->nms_eta, sizeof(float));
  cnrtKernelParamsBufferAddParam(
      params, &nms_param->background_label, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &nms_param->batch_size, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &nms_param->class_num, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &nms_param->num_boxes, sizeof(int));
  cnrtKernelParamsBufferAddParam(params, &nms_param->box_size, sizeof(int));
  // cnrtKernelParamsBufferAddParam(
  //     params, &nms_param->work_space, sizeof(void *));
  cnrtKernelParamsBufferMarkStatic(params);
  cnrtKernelParamsBufferAddParam(params, &data_type, sizeof(DataType));

  cnmlTensor_t input_tensors[2];
  input_tensors[0] = bboxes;
  input_tensors[1] = scores;
  cnmlTensor_t output_tensors[2];
  output_tensors[0] = outs;
  output_tensors[1] = num_outs;
  cnmlTensor_t static_tensors[1];
  static_tensors[0] = workspace_tensor;

  cnmlCreatePluginOp(op_ptr,
                     "multiclass_nms_paddle",
                     reinterpret_cast<void *>(multiclass_nms_paddle_entry),
                     params,
                     input_tensors,
                     2,
                     output_tensors,
                     2,
                     static_tensors,
                     1);

  cnrtDestroyKernelParamsBuffer(params);

  return 0;
}
