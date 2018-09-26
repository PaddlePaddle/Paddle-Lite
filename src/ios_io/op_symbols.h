/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License. */

#pragma once

#include "operators/batchnorm_op.h"
#include "operators/bilinear_interp_op.h"
#include "operators/box_coder_op.h"
#include "operators/concat_op.h"
#include "operators/conv_op.h"
#include "operators/conv_transpose_op.h"
#include "operators/crf_op.h"
#include "operators/depthwise_conv_op.h"
#include "operators/dropout_op.h"
#include "operators/elementwise_add_op.h"
#include "operators/feed_op.h"
#include "operators/fetch_op.h"
#include "operators/flatten_op.h"
#include "operators/fusion_conv_add.h"
#include "operators/fusion_conv_add_add_prelu_op.h"
#include "operators/fusion_conv_add_bn_op.h"
#include "operators/fusion_conv_add_bn_relu_op.h"
#include "operators/fusion_conv_add_prelu_op.h"
#include "operators/fusion_conv_add_relu_op.h"
#include "operators/fusion_conv_bn_add_relu_op.h"
#include "operators/fusion_conv_bn_relu_op.h"
#include "operators/fusion_dwconv_bn_relu_op.h"
#include "operators/fusion_elementwise_add_relu_op.h"
#include "operators/fusion_fc_op.h"
#include "operators/fusion_fc_relu_op.h"
#include "operators/gru_op.h"
#include "operators/im2sequence_op.h"
#include "operators/lookup_op.h"
#include "operators/lrn_op.h"
#include "operators/mul_op.h"
#include "operators/multiclass_nms_op.h"
#include "operators/pool_op.h"
#include "operators/prelu_op.h"
#include "operators/prior_box_op.h"
#include "operators/relu_op.h"
#include "operators/reshape_op.h"
#include "operators/resize_op.h"
#include "operators/scale_op.h"
#include "operators/shape_op.h"
#include "operators/sigmoid_op.h"
#include "operators/slice_op.h"
#include "operators/softmax_op.h"
#include "operators/split_op.h"
#include "operators/transpose_op.h"
