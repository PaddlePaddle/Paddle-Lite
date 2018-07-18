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

#include "common/types.h"
#include <vector>

namespace paddle_mobile {

const std::string G_OP_TYPE_CONV = "conv2d";
const std::string G_OP_TYPE_BATCHNORM = "batch_norm";
const std::string G_OP_TYPE_BOX_CODER = "box_coder";
const std::string G_OP_TYPE_CONCAT = "concat";
const std::string G_OP_TYPE_ELEMENTWISE_ADD = "elementwise_add";
const std::string G_OP_TYPE_FUSION_CONV_ADD_RELU = "fusion_conv_add_relu";
const std::string G_OP_TYPE_FUSION_CONV_ADD_BN_RELU = "fusion_conv_add_bn_relu";
const std::string G_OP_TYPE_FUSION_DWCONV_BN_RELU = "fusion_dwconv_bn_relu";
const std::string G_OP_TYPE_FUSION_CONV_BN_RELU = "fusion_conv_bn_relu";
const std::string G_OP_TYPE_FC = "fusion_fc";
const std::string G_OP_TYPE_FUSION_CONV_ADD = "fusion_conv_add";
const std::string G_OP_TYPE_LRN = "lrn";
const std::string G_OP_TYPE_MUL = "mul";
const std::string G_OP_TYPE_MULTICLASS_NMS = "multiclass_nms";
const std::string G_OP_TYPE_POOL2D = "pool2d";
const std::string G_OP_TYPE_PRIOR_BOX = "prior_box";
const std::string G_OP_TYPE_RELU = "relu";
const std::string G_OP_TYPE_RESHAPE = "reshape";
const std::string G_OP_TYPE_SIGMOID = "sigmoid";
const std::string G_OP_TYPE_SOFTMAX = "softmax";
const std::string G_OP_TYPE_TRANSPOSE = "transpose";
const std::string G_OP_TYPE_SPLIT = "split";
const std::string G_OP_TYPE_FEED = "feed";
const std::string G_OP_TYPE_FETCH = "fetch";
const std::string G_OP_TYPE_DEPTHWISE_CONV = "depthwise_conv2d";
const std::string G_OP_TYPE_IM2SEQUENCE = "im2sequence";
const std::string G_OP_TYPE_DROPOUT = "dropout";

std::unordered_map<
    std::string, std::pair<std::vector<std::string>, std::vector<std::string>>>
    op_input_output_key = {
        {G_OP_TYPE_CONV, {{"Input"}, {"Output"}}},
        {G_OP_TYPE_FUSION_DWCONV_BN_RELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_FUSION_CONV_BN_RELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_FUSION_CONV_ADD, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_RELU, {{"X"}, {"Out"}}},
        {G_OP_TYPE_SOFTMAX, {{"X"}, {"Out"}}},
        {G_OP_TYPE_MUL, {{"X"}, {"Out"}}},
        {G_OP_TYPE_ELEMENTWISE_ADD, {{"X", "Y"}, {"Out"}}},
        {G_OP_TYPE_POOL2D, {{"X"}, {"Out"}}},
        {G_OP_TYPE_BATCHNORM, {{"X"}, {"Y"}}},
        {G_OP_TYPE_LRN, {{"X"}, {"Out"}}},
        {G_OP_TYPE_CONCAT, {{"X"}, {"Out"}}},
        {G_OP_TYPE_SPLIT, {{"X"}, {"Out"}}},
        {G_OP_TYPE_FEED, {{"X"}, {"Out"}}},
        {G_OP_TYPE_FETCH, {{"X"}, {"Out"}}},
        {G_OP_TYPE_TRANSPOSE, {{"X"}, {"Out"}}},
        {G_OP_TYPE_BOX_CODER,
         {{"PriorBox", "PriorBoxVar", "TargetBox"}, {"OutputBox"}}},
        {G_OP_TYPE_FUSION_CONV_ADD_BN_RELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_PRIOR_BOX, {{"Image", "Input"}, {"Boxes", "Variances"}}},
        {G_OP_TYPE_MULTICLASS_NMS, {{"BBoxes", "Scores"}, {"Out"}}},
        {G_OP_TYPE_FC, {{"X", "Y", "Z"}, {"Out"}}},
        {G_OP_TYPE_RESHAPE, {{"X"}, {"Out"}}},
        {G_OP_TYPE_DEPTHWISE_CONV, {{"Input"}, {"Output"}}},
        {G_OP_TYPE_FUSION_CONV_ADD_RELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_IM2SEQUENCE, {{"X"}, {"Out"}}},
        {G_OP_TYPE_DROPOUT, {{"X"}, {"Out"}}}};

}  // namespace paddle_mobile
