// Copyright (c) 2020 smarsu. All Rights Reserved.

#ifndef LITE_KERNELS_MLU_BRIDGES_MULTICLASS_NMS_API_H_
#define LITE_KERNELS_MLU_BRIDGES_MULTICLASS_NMS_API_H_

// #define ALIGN_UP(a, b) (((a) + (b) - 1) / (b) * (b))
// #define ALIGN_DN(a, b) ((a) / (b) * (b))
// #define DIV_UP(a, b) (((a) + (b) - 1) / (b))
// #define DIV_DN(a, b) ((a) / (b))

// #define MAX(a, b) ((a) >= (b) ? (a) : (b))
// #define MIN(a, b) ((a) <= (b) ? (a) : (b))
// #define ABS(a) (((a) > 0) ? (a) : (-(a)))

// #define INIFITE 0x7F800000
#include <cnml.h>
#include <cnrt.h>

enum DataType {
  kInvalid,
  kFloat32,
  kFloat16,
  kUint8,
  kInt8,
  kInt16,
  kInt32,
};

enum TopkSplitStrategy {
  kAuto,
  kSplitN,
  kSplitC,
};

enum ColorType {
  kGray,
  kRGB,
  kBGR,
  kRGBA,
};

struct multiclass_nms_param {
  float score_threshold;
  int nms_top_k;
  int keep_top_k;
  float nms_threshold;
  bool normalized;
  float nms_eta;
  int background_label;
  int batch_size;
  int class_num;
  int num_boxes;
  int box_size;
};

typedef struct multiclass_nms_param *multiclass_nms_param_t;

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
                                 int box_size);

void destory_multiclass_nms_param(multiclass_nms_param_t *params);

int create_multiclass_nms_op(cnmlBaseOp_t *op_ptr,
                             multiclass_nms_param_t nms_param,
                             cnmlTensor_t bboxes,
                             cnmlTensor_t scores,
                             cnmlTensor_t outs,
                             cnmlTensor_t num_outs,
                             cnmlTensor_t workspace_tensor,
                             bool float_precision);

#endif  // LITE_KERNELS_MLU_BRIDGES_MULTICLASS_NMS_API_H_
