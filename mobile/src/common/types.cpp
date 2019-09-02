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

const char *G_OP_TYPE_CONV = "conv2d";
const char *G_OP_TYPE_BATCHNORM = "batch_norm";
const char *G_OP_TYPE_INSTANCENORM = "instance_norm";
const char *G_OP_TYPE_BOX_CODER = "box_coder";
const char *G_OP_TYPE_CONCAT = "concat";
const char *G_OP_TYPE_ELEMENTWISE_ADD = "elementwise_add";
const char *G_OP_TYPE_ELEMENTWISE_SUB = "elementwise_sub";
const char *G_OP_TYPE_ELEMENTWISE_MUL = "elementwise_mul";
const char *G_OP_TYPE_FILL_CONSTANT = "fill_constant";
const char *G_OP_TYPE_FUSION_CONV_ADD_RELU = "fusion_conv_add_relu";
const char *G_OP_TYPE_FUSION_CONV_ADD_PRELU = "fusion_conv_add_prelu";
const char *G_OP_TYPE_FUSION_CONV_ADD_ADD_PRELU = "fusion_conv_add_add_prelu";
const char *G_OP_TYPE_FUSION_CONV_ADD_BN_RELU = "fusion_conv_add_bn_relu";
const char *G_OP_TYPE_FUSION_CONV_BN_ADD_RELU = "fusion_conv_bn_add_relu";
const char *G_OP_TYPE_FUSION_DWCONV_BN_RELU = "fusion_dwconv_bn_relu";
const char *G_OP_TYPE_FUSION_CONV_RELU = "fusion_conv_relu";
const char *G_OP_TYPE_FUSION_CONV_BN_RELU = "fusion_conv_bn_relu";
const char *G_OP_TYPE_FC = "fusion_fc";
const char *G_OP_TYPE_FUSION_CONV_ADD = "fusion_conv_add";
const char *G_OP_TYPE_LRN = "lrn";
const char *G_OP_TYPE_MUL = "mul";
const char *G_OP_TYPE_MULTICLASS_NMS = "multiclass_nms";
const char *G_OP_TYPE_NORM = "norm";
const char *G_OP_TYPE_POLYGON_BOX_TRANSFORM = "polygon_box_transform";
const char *G_OP_TYPE_POOL2D = "pool2d";
const char *G_OP_TYPE_PRIOR_BOX = "prior_box";
const char *G_OP_TYPE_DENSITY_PRIOR_BOX = "density_prior_box";
const char *G_OP_TYPE_RELU = "relu";
const char *G_OP_TYPE_RELU6 = "relu6";
const char *G_OP_TYPE_LEAKY_RELU = "leaky_relu";
const char *G_OP_TYPE_RESHAPE = "reshape";
const char *G_OP_TYPE_RESHAPE2 = "reshape2";
const char *G_OP_TYPE_SCALE = "scale";
const char *G_OP_TYPE_SIGMOID = "sigmoid";
const char *G_OP_TYPE_SOFTMAX = "softmax";
const char *G_OP_TYPE_TRANSPOSE = "transpose";
const char *G_OP_TYPE_TRANSPOSE2 = "transpose2";
const char *G_OP_TYPE_SPLIT = "split";
const char *G_OP_TYPE_FEED = "feed";
const char *G_OP_TYPE_FETCH = "fetch";
const char *G_OP_TYPE_DEPTHWISE_CONV = "depthwise_conv2d";
const char *G_OP_TYPE_IM2SEQUENCE = "im2sequence";
const char *G_OP_TYPE_DROPOUT = "dropout";
const char *G_OP_TYPE_FUSION_CONV_ADD_BN = "fusion_conv_add_bn";
const char *G_OP_TYPE_FUSION_POOL_BN = "fusion_pool_bn";
const char *G_OP_TYPE_FUSION_ELEMENTWISE_ADD_RELU =
    "fusion_elementwise_add_relu";
const char *G_OP_TYPE_FUSION_FC_RELU = "fusion_fc_relu";
const char *G_OP_TYPE_REGION = "region";
const char *G_OP_TYPE_FUSION_CONV_BN = "fusion_conv_bn";
const char *G_OP_TYPE_CONV_TRANSPOSE = "conv2d_transpose";
const char *G_OP_TYPE_PRELU = "prelu";
const char *G_OP_TYPE_LOOKUP_TABLE = "lookup_table";
const char *G_OP_TYPE_GRU = "gru";
const char *G_OP_TYPE_GRU_UNIT = "gru_unit";
const char *G_OP_TYPE_CRF = "crf_decoding";
const char *G_OP_TYPE_BILINEAR_INTERP = "bilinear_interp";
const char *G_OP_TYPE_NEAREST_INTERP = "nearest_interp";
const char *G_OP_TYPE_FLATTEN = "flatten";
const char *G_OP_TYPE_FLATTEN2 = "flatten2";
const char *G_OP_TYPE_SHAPE = "shape";
const char *G_OP_TYPE_SUM = "sum";
const char *G_OP_TYPE_TOP_K = "top_k";
const char *G_OP_TYPE_CAST = "cast";
const char *G_OP_TYPE_LOG = "log";
const char *G_OP_TYPE_LOD_RESET = "lod_reset";
const char *G_OP_TYPE_LESS_THAN = "less_than";
const char *G_OP_TYPE_LOGICAL_AND = "logical_and";
const char *G_OP_TYPE_LOGICAL_OR = "logical_or";
const char *G_OP_TYPE_LOGICAL_NOT = "logical_not";
const char *G_OP_TYPE_LOGICAL_XOR = "logical_xor";
const char *G_OP_TYPE_WRITE_TO_ARRAY = "write_to_array";
const char *G_OP_TYPE_READ_FROM_ARRAY = "read_from_array";
const char *G_OP_TYPE_IS_EMPTY = "is_empty";
const char *G_OP_TYPE_INCREMENT = "increment";
const char *G_OP_TYPE_EXP = "exp";

const char *G_OP_TYPE_QUANTIZE = "quantize";
const char *G_OP_TYPE_DEQUANTIZE = "dequantize";
const char *G_OP_TYPE_FUSION_DEQUANT_BN = "fusion_dequant_bn";
const char *G_OP_TYPE_FUSION_DEQUANT_ADD_BN = "fusion_dequant_add_bn";
const char *G_OP_TYPE_FUSION_DEQUANT_BN_RELU = "fusion_dequant_bn_relu";
const char *G_OP_TYPE_FUSION_DEQUANT_ADD_BN_RELU = "fusion_dequant_add_bn_relu";
const char *G_OP_TYPE_FUSION_DEQUANT_ADD_BN_QUANT =
    "fusion_dequant_add_bn_quant";
const char *G_OP_TYPE_FUSION_DEQUANT_ADD_BN_RELU_QUANT =
    "fusion_dequant_add_bn_relu_quant";

const char *G_OP_TYPE_TANH = "tanh";
const char *G_OP_TYPE_FUSION_DECONV_RELU = "fusion_deconv_relu";
const char *G_OP_TYPE_FUSION_DECONV_ADD = "fusion_deconv_add";
const char *G_OP_TYPE_FUSION_DECONV_ADD_RELU = "fusion_deconv_add_relu";

const char *G_OP_TYPE_SEQUENCE_EXPAND = "sequence_expand";
const char *G_OP_TYPE_SEQUENCE_POOL = "sequence_pool";
const char *G_OP_TYPE_SEQUENCE_SOFTMAX = "sequence_softmax";
const char *G_OP_TYPE_SLICE = "slice";
const char *G_OP_TYPE_ANCHOR_GENERATOR = "anchor_generator";
const char *G_OP_TYPE_GENERATE_PROPOSALS = "generate_proposals";
const char *G_OP_TYPE_PSROI_POOL = "psroi_pool";
const char *G_OP_TYPE_ROIALIGN_POOL = "roialign_pool";
const char *G_OP_TYPE_ROI_PERSPECTIVE = "roi_perspective_transform";
const char *G_OP_TYPE_PAD2D = "pad2d";
const char *G_OP_TYPE_FUSION_DECONV_ADD_BN_RELU = "fusion_deconv_add_bn_relu";
const char *G_OP_TYPE_FUSION_DECONV_ADD_BN = "fusion_deconv_add_bn";
const char *G_OP_TYPE_FUSION_DECONV_BN_RELU = "fusion_deconv_bn_relu";
const char *G_OP_TYPE_ASSIGN = "assign";
const char *G_OP_TYPE_REDUCE_PROD = "reduce_prod";
const char *G_OP_TYPE_EQUAL = "equal";
const char *G_OP_TYPE_CONDITIONAL_BLOCK = "conditional_block";
const char *G_OP_TYPE_RANGE = "range";
const char *G_OP_TYPE_WHILE = "while";
const char *G_OP_TYPE_BEAM_SEARCH_DECODE = "beam_search_decode";
const char *G_OP_TYPE_FILL_CONSTAN_BATCH_SIZE_LIKE =
    "fill_constant_batch_size_like";

std::unordered_map<
    std::string, std::pair<std::vector<std::string>, std::vector<std::string>>>
    op_input_output_key = {
        {G_OP_TYPE_CONV, {{"Input"}, {"Output"}}},
        {G_OP_TYPE_FUSION_DWCONV_BN_RELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_FUSION_CONV_RELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_FUSION_CONV_BN_RELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_PRELU, {{"X", "Alpha"}, {"Out"}}},
        {G_OP_TYPE_FUSION_CONV_ADD, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_RELU, {{"X"}, {"Out"}}},
        {G_OP_TYPE_RELU6, {{"X"}, {"Out"}}},
        {G_OP_TYPE_LEAKY_RELU, {{"X"}, {"Out"}}},
        {G_OP_TYPE_SCALE, {{"X"}, {"Out"}}},
        {G_OP_TYPE_SOFTMAX, {{"X"}, {"Out"}}},
        {G_OP_TYPE_SIGMOID, {{"X"}, {"Out"}}},
        {G_OP_TYPE_MUL, {{"X"}, {"Out"}}},
        {G_OP_TYPE_ELEMENTWISE_ADD, {{"X", "Y"}, {"Out"}}},
        {G_OP_TYPE_ELEMENTWISE_SUB, {{"X", "Y"}, {"Out"}}},
        {G_OP_TYPE_ELEMENTWISE_MUL, {{"X", "Y"}, {"Out"}}},
        {G_OP_TYPE_POOL2D, {{"X"}, {"Out"}}},
        {G_OP_TYPE_BATCHNORM, {{"X"}, {"Y"}}},
        {G_OP_TYPE_INSTANCENORM, {{"X"}, {"Out"}}},
        {G_OP_TYPE_LRN, {{"X"}, {"Out"}}},
        {G_OP_TYPE_CONCAT, {{"X"}, {"Out"}}},
        {G_OP_TYPE_SPLIT, {{"X"}, {"Out"}}},
        {G_OP_TYPE_FEED, {{"X"}, {"Out"}}},
        {G_OP_TYPE_FETCH, {{"X"}, {"Out"}}},
        {G_OP_TYPE_TRANSPOSE, {{"X"}, {"Out"}}},
        {G_OP_TYPE_TRANSPOSE2, {{"X"}, {"Out", "XShape"}}},
        {G_OP_TYPE_BOX_CODER,
         {{"PriorBox", "PriorBoxVar", "TargetBox"}, {"OutputBox"}}},
        {G_OP_TYPE_FUSION_CONV_ADD_BN_RELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_FUSION_CONV_BN_ADD_RELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_PRIOR_BOX, {{"Image", "Input"}, {"Boxes", "Variances"}}},
        {G_OP_TYPE_DENSITY_PRIOR_BOX,
         {{"Image", "Input"}, {"Boxes", "Variances"}}},
        {G_OP_TYPE_MULTICLASS_NMS, {{"BBoxes", "Scores"}, {"Out"}}},
        {G_OP_TYPE_POLYGON_BOX_TRANSFORM, {{"Input"}, {"Output"}}},
        {G_OP_TYPE_FC, {{"X", "Y", "Z"}, {"Out"}}},
        {G_OP_TYPE_RESHAPE, {{"X"}, {"Out"}}},
        {G_OP_TYPE_RESHAPE2, {{"X"}, {"Out", "XShape"}}},
        {G_OP_TYPE_DEPTHWISE_CONV, {{"Input"}, {"Output"}}},
        {G_OP_TYPE_FILL_CONSTANT, {{}, {"Out"}}},
        {G_OP_TYPE_FUSION_CONV_ADD_RELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_FUSION_CONV_ADD_PRELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_FUSION_CONV_ADD_ADD_PRELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_IM2SEQUENCE, {{"X"}, {"Out"}}},
        {G_OP_TYPE_DROPOUT, {{"X"}, {"Out"}}},
        {G_OP_TYPE_EXP, {{"X"}, {"Out"}}},
        {G_OP_TYPE_FUSION_CONV_ADD_BN, {{"Input"}, {"Y"}}},
        {G_OP_TYPE_FUSION_POOL_BN, {{"X"}, {"Y"}}},
        {G_OP_TYPE_FUSION_ELEMENTWISE_ADD_RELU, {{"X", "Y"}, {"Out"}}},
        {G_OP_TYPE_FUSION_FC_RELU, {{"X", "Y", "Z"}, {"Out"}}},
        {G_OP_TYPE_REGION, {{"X"}, {"Out"}}},
        {G_OP_TYPE_FUSION_CONV_BN, {{"Input"}, {"Y"}}},
        {G_OP_TYPE_LOOKUP_TABLE, {{"W", "Ids"}, {"Out"}}},
        {G_OP_TYPE_GRU,
         {{"Input", "H0", "Weight", "Bias"},
          {"BatchGate", "BatchResetHiddenPrev", "BatchHidden", "Hidden"}}},
        {G_OP_TYPE_GRU_UNIT,
         {{"Input", "HiddenPrev", "Weight", "Bias"},
          {"Gate", "ResetHiddenPrev", "Hidden"}}},
        {G_OP_TYPE_CRF, {{"Emission", "Transition", "Label"}, {"ViterbiPath"}}},
        {G_OP_TYPE_BILINEAR_INTERP, {{"OutSize", "X"}, {"Out"}}},
        {G_OP_TYPE_NEAREST_INTERP, {{"OutSize", "X"}, {"Out"}}},
        {G_OP_TYPE_FLATTEN, {{"X"}, {"Out"}}},
        {G_OP_TYPE_FLATTEN2, {{"X"}, {"Out"}}},
        {G_OP_TYPE_SHAPE, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_CONV_TRANSPOSE, {{"Input"}, {"Output"}}},
        {G_OP_TYPE_SUM, {{"X"}, {"Out"}}},
        {G_OP_TYPE_TOP_K, {{"X"}, {"Out", "Indices"}}},
        {G_OP_TYPE_CAST, {{"X"}, {"Out"}}},
        {G_OP_TYPE_QUANTIZE, {{"X"}, {"Out", "OutScale"}}},
        {G_OP_TYPE_DEQUANTIZE, {{"X", "Scale"}, {"Out"}}},
        {G_OP_TYPE_FUSION_DEQUANT_BN, {{"X", "Scale"}, {"Out"}}},
        {G_OP_TYPE_FUSION_DEQUANT_ADD_BN, {{"X", "Scale"}, {"Out"}}},
        {G_OP_TYPE_FUSION_DEQUANT_BN_RELU, {{"X", "Scale"}, {"Out"}}},
        {G_OP_TYPE_FUSION_DEQUANT_ADD_BN_RELU, {{"X", "Scale"}, {"Out"}}},
        {G_OP_TYPE_FUSION_DEQUANT_ADD_BN_RELU_QUANT,
         {{"X", "Scale"}, {"Out", "OutScale"}}},
        {G_OP_TYPE_FUSION_DEQUANT_ADD_BN_QUANT,
         {{"X", "Scale"}, {"Out", "OutScale"}}},
        {G_OP_TYPE_TANH, {{"X"}, {"Out"}}},
        {G_OP_TYPE_FUSION_DECONV_RELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_FUSION_DECONV_ADD, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_FUSION_DECONV_ADD_RELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_SEQUENCE_EXPAND, {{"X", "Y"}, {"Out"}}},
        {G_OP_TYPE_SEQUENCE_POOL, {{"X"}, {"Out"}}},
        {G_OP_TYPE_SEQUENCE_SOFTMAX, {{"X"}, {"Out"}}},
        {G_OP_TYPE_NORM, {{"X"}, {"Out", "Norm"}}},
        {G_OP_TYPE_LOG, {{"X"}, {"Out"}}},
        {G_OP_TYPE_LOD_RESET, {{"X", "Y"}, {"Out"}}},
        {G_OP_TYPE_LESS_THAN, {{"X", "Y"}, {"Out"}}},
        {G_OP_TYPE_LOGICAL_AND, {{"X", "Y"}, {"Out"}}},
        {G_OP_TYPE_LOGICAL_OR, {{"X", "Y"}, {"Out"}}},
        {G_OP_TYPE_LOGICAL_XOR, {{"X", "Y"}, {"Out"}}},
        {G_OP_TYPE_LOGICAL_NOT, {{"X"}, {"Out"}}},
        {G_OP_TYPE_WRITE_TO_ARRAY, {{"X", "I"}, {"Out"}}},
        {G_OP_TYPE_READ_FROM_ARRAY, {{"X", "I"}, {"Out"}}},
        {G_OP_TYPE_IS_EMPTY, {{"X"}, {"Out"}}},
        {G_OP_TYPE_INCREMENT, {{"X"}, {"Out"}}},
        {G_OP_TYPE_SLICE, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_ANCHOR_GENERATOR, {{"Input"}, {"Anchors", "Variances"}}},
        {G_OP_TYPE_GENERATE_PROPOSALS,
         {{"Scores", "BboxDeltas", "ImInfo", "Anchors", "Variances"},
          {"RpnRois", "RpnRoiProbs"}}},
        {G_OP_TYPE_PSROI_POOL, {{"X", "ROIs"}, {"Out"}}},
        {G_OP_TYPE_ROIALIGN_POOL, {{"X", "ROIs"}, {"Out"}}},
        {G_OP_TYPE_ROI_PERSPECTIVE, {{"X", "ROIs"}, {"Out"}}},
        {G_OP_TYPE_FUSION_DECONV_ADD_BN_RELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_FUSION_DECONV_ADD_BN, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_FUSION_DECONV_BN_RELU, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_REDUCE_PROD, {{"X"}, {"Out"}}},
        {G_OP_TYPE_ASSIGN, {{"X"}, {"Out"}}},
        {G_OP_TYPE_EQUAL, {{"X", "Y"}, {"Out"}}},
        {G_OP_TYPE_RANGE, {{"Start", "End", "Step"}, {"Out"}}},
        {G_OP_TYPE_CONDITIONAL_BLOCK, {{"Input", "Cond"}, {"Out", "Scope"}}},
        {G_OP_TYPE_WHILE, {{"Condition", "X"}, {"Out", "StepScopes"}}},
        {G_OP_TYPE_BEAM_SEARCH_DECODE,
         {{"Ids", "Scores"}, {"SentenceIds", "SentenceScores"}}},
        {G_OP_TYPE_FILL_CONSTAN_BATCH_SIZE_LIKE, {{"Input"}, {"Out"}}},
        {G_OP_TYPE_PAD2D, {{"X"}, {"Out"}}}};
}  // namespace paddle_mobile
