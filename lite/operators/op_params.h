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

#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "lite/api/paddle_place.h"
#include "lite/core/model/base/apis.h"
#include "lite/core/scope.h"
#include "lite/core/tensor.h"
#include "lite/core/types.h"
#include "lite/model_parser/cpp_desc.h"
#include "lite/utils/all.h"
/*
 * This file contains all the argument parameter data structure for operators.
 */

namespace paddle {
namespace lite {
namespace operators {

struct ParamBase {};

using param_t = Any;
#define WITH_INT8_CONFIG             \
  bool enable_int8{false};           \
  float input_scale{1.0f};           \
  std::vector<float> weight_scale{}; \
  float output_scale{1.0f};          \
  int bit_length{8};

/// ----------------------- Functional operators ------------------------------
struct FeedParam : ParamBase {
  std::vector<lite::Tensor>* feed_list{};
  lite::Tensor* out{};
  int col;
};

struct FetchParam : ParamBase {
  const lite::Tensor* input{};
  std::vector<lite::Tensor>* fetch_list{};
  int col;
};

// Helper op for lite framework
struct IoCopyParam : ParamBase {
  const lite::Tensor* x{nullptr};
  const std::vector<lite::Tensor>* x_array{nullptr};
  lite::Tensor* y{nullptr};
  std::vector<lite::Tensor>* y_array{nullptr};
  int process_type{0};
};

struct LayoutParam : ParamBase {
  const lite::Tensor* x{};
  lite::Tensor* y{};
  int process_type{0};
};

struct CalibParam : ParamBase {
  const lite::Tensor* input{};
  lite::Tensor* output{};
  float scale;
};

struct CalibInplaceParam : ParamBase {
  lite::Tensor* input{};
  lite::Tensor* output{};
  float scale;
};

struct SubgraphParam : ParamBase {
  std::vector<std::string> input_names{};
  std::vector<std::string> output_names{};
  std::vector<std::string> input_data_names{};
  std::vector<std::string> output_data_names{};
  std::vector<float> input_data_scales{};
  std::vector<float> output_data_scales{};
  int block_idx{-1};
  std::shared_ptr<const cpp::ProgramDesc> program_desc{nullptr};
  Scope* exec_scope{nullptr};
};

/// -------------------------- NN operators ------------------------------------

struct FcParam : ParamBase {
  lite::Tensor* input{nullptr};
  lite::Tensor* w{nullptr};
  lite::Tensor* bias{nullptr};
  lite::Tensor* Prelu_alpha{nullptr};
  lite::Tensor* output{nullptr};
  lite::DDim in_mat_dims;
  // original dims of input weight
  lite::DDim w_dims;
  int in_num_col_dims{1};
  std::string activation_type{""};
  bool padding_weights{false};
  std::string Prelu_mode{
      "channel"};  // prelu param, can be "all", "channel" or "element"
  std::string op_type{"mul"};
  float alpha{6.f};
  // for int8
  WITH_INT8_CONFIG
};

struct SearchSeqFcParam : ParamBase {
  lite::Tensor* x{nullptr};
  lite::Tensor* w{nullptr};
  lite::Tensor* b{nullptr};
  lite::Tensor* out{nullptr};
  int out_size;
};

// For Interpolate Op
struct InterpolateParam : ParamBase {
  lite::Tensor* X{};
  lite::Tensor* OutSize{};
  lite::Tensor* Out{};
  std::vector<const lite::Tensor*> SizeTensor;
  lite::Tensor* Scale{};

  float scale{0.f};
  std::vector<float> scale_v{};
  int out_h{-1};
  int out_w{-1};
  bool align_corners{true};
  int align_mode{1};
  bool version_2{false};
  std::string interp_method{"Nearest"};
  DataLayoutType data_layout{DATALAYOUT(kNCHW)};
};

// For Mul Op
struct MulParam : ParamBase {
  const lite::Tensor* x{};
  const lite::Tensor* y{};
  lite::Tensor* output{};

  int x_num_col_dims{1};
  int y_num_col_dims{1};
  // for int8
  WITH_INT8_CONFIG
};

struct MulGradParam : ParamBase {
  const lite::Tensor* x{};
  const lite::Tensor* y{};
  const lite::Tensor* output_grad{};
  lite::Tensor* x_grad{};
  lite::Tensor* y_grad{};

  int x_num_col_dims{1};
  int y_num_col_dims{1};
};

// For Stack Op
struct StackParam : ParamBase {
  std::vector<lite::Tensor*> X;
  lite::Tensor* Out{};

  int axis{0};
};

// For Unstack Op
struct UnstackParam : ParamBase {
  const lite::Tensor* X{nullptr};
  std::vector<lite::Tensor*> Out{};

  int axis{0};
  int num{1};
};

// For Power Op
struct PowerParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};

  float scale{};
  float shift{};
  float power{};
};

// For Pow Op
struct PowParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};

  float factor{1.f};
};

// For Sign Op
struct SignParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct ShuffleChannelParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};

  int group;
};

// For Yolobox
struct YoloBoxParam : ParamBase {
  lite::Tensor* X{};
  lite::Tensor* ImgSize{};
  lite::Tensor* Boxes{};
  lite::Tensor* Scores{};

  std::vector<int> anchors{};
  int class_num{0};
  float conf_thresh{0.f};
  int downsample_ratio{0};
  bool clip_bbox{true};
  float scale_x_y{1.0f};
};

// For Scale Op
struct ScaleParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* output{};

  float scale{1.f};
  float bias{0.f};
  bool bias_after_scale{true};
  std::string activation_type{""};
  bool fuse_relu{false};
  float alpha{6.f};

  bool fuse_scaleact{false};
  float scale1{1.f};
  float bias1{0.f};
};

// For Scatter OP
struct ScatterParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* indexs{};
  lite::Tensor* updates{};
  lite::Tensor* output{};

  bool overwrite{true};
};

// For Softmax op
struct SoftmaxParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* output{};
  int axis{-1};
  bool use_cudnn{true};
  bool eleminate_success{false};
};

// For LogSoftmax op
struct LogSoftmaxParam : ParamBase {
  const lite::Tensor* x{};
  lite::Tensor* output{};
  int axis{-1};
};

// For Reshape and Reshape2 Op
struct ReshapeParam : ParamBase {
  const lite::Tensor* x{};
  std::vector<const lite::Tensor*> shape_tensor_vct{};
  const lite::Tensor* shape_tensor{};
  std::vector<int> shape_vct{};
  lite::Tensor* output{};

  lite::Tensor* xshape{};
  bool inplace{false};

#ifdef LITE_WITH_METAL
  std::vector<int> excepted_transpose_;
#endif
};

// For Concat op
struct ConcatParam : ParamBase {
  std::vector<lite::Tensor*> x{};
  lite::Tensor* output{};
  int axis{0};
  lite::Tensor* axis_tensor{};
};

/// ----------------------- activation operators ----------------------
struct ActivationParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite_api::ActivationType active_type{lite_api::ActivationType::kIndentity};
  bool has_active{false};
  float Leaky_relu_alpha{0.f};   // leaky_relu param
  float Relu_clipped_coef{6.f};  // relu_clipped param
  std::string Prelu_mode{
      "channel"};  // prelu param, can be "all", "channel" or "element"
  lite::Tensor* Prelu_alpha{};  // prelu param
  float Swish_beta;             // swish param
  // hard_sigmoid param
  float hard_sigmoid_slope{0.2f};
  float hard_sigmoid_offset{0.5f};
  // hard_swish param
  float hard_swish_threshold{6.0f};
  float hard_swish_scale{6.0f};
  float hard_swish_offset{3.0f};
  // swish param
  float swish_scale{6.0f};
  // thresholded_relu
  float relu_threshold{1.0f};
  // elu
  float Elu_alpha{1.0f};
  // relu6
  float threshold{6.0f};
  // gelu
  bool gelu_approximate{false};
  // softplus
  float softplus_beta{1.0f};
  float softplus_threshold{20.f};
};

struct ActivationGradParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Out{};
  // for backward
  lite::Tensor* X_grad{};
  const lite::Tensor* Out_grad{};
};

// For Sparse Convolution op
struct SparseConvParam : ParamBase {
  const lite::Tensor* x{};
  // An array of float values storing non-zero kernel elements
  lite::Tensor* nonzero_weights{};
  /* An array of int32_t values storing scaled
   * [by sizeof(input element)] difference  between input channels
   * corresponding to successive non-zero element
   */
  lite::Tensor* diffs{};
  // the number of non-zero kernel elements per each output channel
  lite::Tensor* oc_nonzeros{};
  lite::Tensor* bias{nullptr};
  lite::Tensor* output{};
  int first_ic{0};
  int flag_semi{0};
  std::vector<int> strides{1, 1};
  std::shared_ptr<std::vector<int>> paddings;
  int groups{1};
  std::shared_ptr<std::vector<int>> dilations;
  // for activation
  bool fuse_relu{false};
  ActivationParam activation_param;
  bool enable_int8{false};
  float input_scale{1.0f};
  std::vector<float> weight_scale{};
  float output_scale{1.0f};
  int bit_length{8};
};

// For Convolution op
struct ConvParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* filter{};
  lite::Tensor* bias{nullptr};
  lite::Tensor* residualData{nullptr};
  lite::Tensor* second_x{nullptr};
  lite::Tensor* output{};
  std::vector<int> strides{1, 1};
  /* paddings type change
   * from std::vector<int> to std::shared_ptr<std::vector<int>>
   * to support dynamically modify padding
   * let kernel param and operator param Synchronous update
   */
  std::shared_ptr<std::vector<int>> paddings;
  int groups{1};
  /* dilations type change
   * from std::vector<int> to std::shared_ptr<std::vector<int>>
   * to support dynamically modify padding
   * let kernel param and operator param Synchronous update
   */
  std::shared_ptr<std::vector<int>> dilations;
  bool fuse_relu_before_depthwise_conv{false};
  bool use_mkldnn{false};
  bool fuse_relu{false};  // only used in mkldnn kernel
  bool fuse_sigmoid{false};
  bool fuse_tanh{false};
  bool fuse_swish{false};
  bool fuse_exp{false};
  bool fuse_abs{false};
  bool use_quantizer{
      false};  // set true for op that should be quantized, only used for cpu
  bool fuse_residual_connection{false};
  float scale_in{1.0f};           // only used with mkl-dnn int8
  float scale_out{1.0f};          // only used with mkl-dnn int8
  float scale_in_eltwise{1.0f};   // only used with mkl-dnn int8
  float scale_weights{1.0f};      // only used with mkl-dnn int8
  bool force_fp32_output{false};  // only used in mkl-dnn int8
  std::string data_format{"Anylayout"};
  // for activation
  ActivationParam activation_param;
  // for elementwise tree fuse
  std::string fuse_elementwise_op_type{""};
  // support var_length or not
  bool var_length{false};
  // only used in conv_transpose.
  std::vector<int> output_size;
  std::vector<int> output_padding;

#ifdef LITE_WITH_FPGA
  lite::Tensor* scale{nullptr};
  struct StrideInfo {
    bool wd_enable_ = false;
    int wd_offset_ = -1;
    int fuse_idx_ = -1;
    int original_out_channel_ = -1;
    int start_idx_ = 0;
    int end_idx_ = 0;
  };
  StrideInfo stride_info_;
#endif

  // for int8
  WITH_INT8_CONFIG
  // for Conv2d+Scale fusion
  std::string scale_activation_type{""};
};

// For BatchNorm op
struct BatchNormParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* bias{};
  lite::Tensor* scale{};
  lite::Tensor* mean{};
  lite::Tensor* variance{};
  lite::Tensor* y{};
  lite::Tensor* mean_out{};
  lite::Tensor* variance_out{};
  lite::Tensor* saved_mean{};
  lite::Tensor* saved_variance{};
  bool is_test{true};
  bool use_global_stats{false};
  float epsilon;
  float momentum;
  DataLayoutType data_layout{DATALAYOUT(kNCHW)};
};

// For Pooling op
struct PoolParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* output{};
  lite::Tensor* mask{};
  std::string pooling_type{""};
  std::vector<int> ksize{};
  bool global_pooling{
      false};  // if true, knernel size and paddings will be ignored
  std::vector<int> strides{1, 1};
  /* paddings type change
   * from std::vector<int> to std::shared_ptr<std::vector<int>>
   * to support dynamically modify padding
   * let kernel param and operator param Synchronous update
   */
  std::shared_ptr<std::vector<int>> paddings;
  bool exclusive{true};
  bool adaptive{false};
  bool ceil_mode{false};
  bool use_quantizer{false};
  std::string padding_algorithm{"EXPLICIT"};
  std::string data_format{"AnyLayout"};
  // for int8
  WITH_INT8_CONFIG
#ifdef LITE_WITH_XPU
  bool pad_zero{false};
#endif
};

// For Dropout op
struct DropoutParam : ParamBase {
  const lite::Tensor* x{};
  const lite::Tensor* seed_tensor{};
  lite::Tensor* output{};
  lite::Tensor* mask{};
  float dropout_prob{.5f};
  bool is_test{false};
  bool fix_seed{false};
  int seed{0};
  std::string dropout_implementation{"downgrade_in_infer"};
};

// For PadConstantLike op
struct PadConstantLikeParam : ParamBase {
  const lite::Tensor* x{};
  const lite::Tensor* y{};
  lite::Tensor* output{};
  float pad_value{0.0f};
};

// For Split op
struct SplitParam : ParamBase {
  const lite::Tensor* x{nullptr};
  std::vector<lite::Tensor*> output{};
  const lite::Tensor* axis_tensor{nullptr};
  std::vector<lite::Tensor*> sections_tensor_list{};

  int axis{-1};
  int num{0};
  std::vector<int> sections;
};

struct UnbindParam : ParamBase {
  lite::Tensor* x{};
  std::vector<lite::Tensor*> output{};

  int axis{-1};
};

// For Transpose op
struct TransposeParam : ParamBase {
  const lite::Tensor* x{};
  lite::Tensor* output{};
  lite::Tensor* xshape{};

  std::vector<int> axis;
  bool use_mkldnn{false};
  std::string data_format{"AnyLayout"};
};

struct TrilTriuParam : ParamBase {
  const lite::Tensor* x{nullptr};
  lite::Tensor* out{nullptr};

  int diagonal{0};
  bool lower{true};
};

/// ----------------------- element wise operators ----------------------
struct ElementwiseParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  int axis{-1};  // for broadcasting.
  // for int8
  WITH_INT8_CONFIG
  float x_input_scale{1.0f};
  float y_input_scale{1.0f};
  // fuse ScaleParam
  bool fuse_scale{false};
  float scale{1.f};
  float bias{0.f};
  bool bias_after_scale{true};
  float alpha{6.f};
  std::string activation_type{""};
};

struct ElementwiseGradParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  const lite::Tensor* OutGrad{};
  lite::Tensor* XGrad{};
  lite::Tensor* YGrad{};
  int axis{-1};  // for broadcasting.
};

struct FusionElementwiseActivationParam : public ElementwiseParam {
  std::string act_type;
};

struct FusionElementwiseActivationGradParam : public ElementwiseGradParam {
  std::string act_type;
};

/// ----------------------- mean operators ----------------------
struct MeanParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct MeanGradParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Out_grad{};
  // for backward
  lite::Tensor* X_grad{};
};

struct FillAnyLikeParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  float value{0.0f};
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
};

/// ----------------------- fill_constant operators ----------------------
struct FillConstantParam : ParamBase {
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
  std::vector<int64_t> shape{};
  lite::Tensor* shape_tensor{nullptr};
  lite::Tensor* value_tensor{nullptr};
  std::vector<lite::Tensor*> shape_tensor_list{};

  float value{0.0f};
  // useless for x86, keep it for compatibility
  bool force_cpu{false};
  lite::Tensor* in{};
  lite::Tensor* out{};
};

struct FillConstantBatchSizeLikeParam : ParamBase {
  const lite::Tensor* input{nullptr};
  lite::Tensor* out{nullptr};

  std::vector<int> shape{};
  int input_dim_idx{0};
  int output_dim_idx{0};
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
  float value{0.0f};
  // useless for x86, keep it for compatibility
  bool force_cpu{false};
};

//
struct FakeQuantizeMovingAvgMaxAbsParam : ParamBase {
  const lite::Tensor* x{};
  const lite::Tensor* in_scale{};
  const lite::Tensor* in_accum{};
  const lite::Tensor* in_state{};
  lite::Tensor* out{};
  lite::Tensor* out_scale{};
  lite::Tensor* out_state{};
  lite::Tensor* out_accum{};
  int bit_length;
  bool is_test{true};
  float moving_rate{0.9f};
};

struct FakeDequantizeMaxAbsParam : ParamBase {
  const lite::Tensor* x{};
  const lite::Tensor* in_scale{};
  lite::Tensor* out{};
  float max_range;
};

struct FakeChannelWiseDequantizeMaxAbsParam : ParamBase {
  const lite::Tensor* x{};
  std::vector<const lite::Tensor*> scale_tensors{};
  lite::Tensor* out{};
  std::vector<int> quant_bits;
};

struct FakeQuantDequantAbsMaxParam : ParamBase {
  const lite::Tensor* x{};
  lite::Tensor* out{};
  lite::Tensor* out_scale{};
  int bit_length;
};

struct FakeChannelWiseQuantDequantAbsMaxParam : ParamBase {
  const lite::Tensor* x{};
  lite::Tensor* out{};
  lite::Tensor* out_scale{};
  int quant_axis;
  int bit_length;
};

struct QuantizeLinearParam : ParamBase {
  const lite::Tensor* x{};
  const lite::Tensor* scale{};
  const lite::Tensor* zero_point{};
  lite::Tensor* y{};
  int quant_axis;
  int bit_length;
};

/// ----------------------- sgd operators ----------------------
struct SGDParam : ParamBase {
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};

  const lite::Tensor* Param{};
  const lite::Tensor* LearningRate{};
  const lite::Tensor* Grad{};
  lite::Tensor* ParamOut{};
};

/// ----------------------- uniform_random operators ----------------------
struct UniformRandomParam : ParamBase {
  const lite::Tensor* shape_tensor{nullptr};
  std::vector<lite::Tensor*> shape_tensor_list{};
  std::vector<int64_t> shape{};
  float min{-1.0f};
  float max{1.0f};
  int seed{0};
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
  lite::Tensor* Out{};
};
/// ----------------------- unfold operators ----------------------
struct UnfoldParam : ParamBase {
  const lite::Tensor* X{nullptr};
  std::vector<int> kernel_sizes{};
  std::vector<int> strides{};
  std::vector<int> paddings{};
  std::vector<int> dilations{};
  lite::Tensor* Y{nullptr};
};
/// ----------------------- negative operators --------------
struct NegativeParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};
/// ----------------------- pad2d operators ----------------------
struct Pad2dParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  std::vector<int> paddings{0, 0, 0, 0};
  std::string mode{"constant"};
  float pad_value = 0.f;
  std::string data_format{"NCHW"};
};

/// ----------------------- Crop operators ----------------------
struct CropParam : ParamBase {
  const lite::Tensor* X{nullptr};
  const lite::Tensor* Y{nullptr};
  const lite::Tensor* Offsets{nullptr};
  lite::Tensor* Out{nullptr};
  std::vector<int> offsets;
  std::vector<int> shape;
};

/// ----------------------- CropTensor operators ----------------------
struct CropTensorParam : ParamBase {
  const lite::Tensor* X{nullptr};
  const lite::Tensor* Shape{nullptr};
  const lite::Tensor* Offsets{nullptr};
  const std::vector<lite::Tensor>* ShapeTensor{nullptr};
  const std::vector<lite::Tensor>* OffsetsTensor{nullptr};
  lite::Tensor* Out{nullptr};
  std::vector<int> offsets;
  std::vector<int> shape;
};

///----------------------- argmax operators ----------------------
struct ArgmaxParam : ParamBase {
  lite::Tensor* X{};
  lite::Tensor* Out{};
  int Axis{0};
  int dtype{-1};
  bool keepdims{false};
};

///----------------------- inverse operators ----------------------
struct InverseParam : ParamBase {
  lite::Tensor* Input{};
  lite::Tensor* Output{};
};

///----------------------- index_select operators ----------------------
struct Index_selectParam : ParamBase {
  lite::Tensor* X{};
  lite::Tensor* Index{};
  lite::Tensor* Out{};
  int dim{0};
};

///----------------------- reverse operators ----------------------
struct ReverseParam : ParamBase {
  lite::Tensor* X{};
  lite::Tensor* Out{};
  // for tensor_array
  std::vector<lite::Tensor>* X_array{nullptr};
  std::vector<lite::Tensor>* Out_array{nullptr};
  std::vector<int> Axis;
};

///----------------------- axpy operators ----------------------
struct AxpyParam : ParamBase {
  lite::Tensor* Scale{};
  lite::Tensor* X{};
  lite::Tensor* Bias{};
  lite::Tensor* Out{};
};
/// ----------------------- GRU unit operators ----------------------f
struct GRUUnitParam : ParamBase {
  enum ActType { identity, sigmoid, tanh, relu };
  const lite::Tensor* input{nullptr};
  const lite::Tensor* hidden_prev{nullptr};
  const lite::Tensor* weight{nullptr};
  const lite::Tensor* bias{nullptr};
  lite::Tensor* gate{nullptr};
  lite::Tensor* reset_hidden_prev{nullptr};
  lite::Tensor* hidden{nullptr};

  int gate_activation{ActType::sigmoid};
  int activation{ActType::tanh};
  bool origin_mode{false};
};

/// ------------------------------ lrn operators ------------------------------
struct LrnParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  int n{5};
  float alpha{1e-4f};
  float beta{0.75f};
  float k{1.f};
  std::string norm_region{"AcrossChannels"};
};

/// ----------------------- decode_bboxes operators ----------------------
struct DecodeBboxesParam : ParamBase {
  const lite::Tensor* loc_data{};
  const lite::Tensor* prior_data{};
  lite::Tensor* bbox_data{};

  int batch_num;
  int num_priors;
  int num_loc_classes{0};
  int background_label_id{0};
  bool share_location{true};
  bool variance_encoded_in_target;
  // code_type:  corner, cente_size, corner_size
  std::string code_type;
};

/// ----------------------- box_coder operators ----------------------
struct BoxCoderParam : ParamBase {
  const lite::Tensor* prior_box{};
  const lite::Tensor* prior_box_var{};
  const lite::Tensor* target_box{};
  lite::Tensor* proposals{};
  // code_type: encode_center_size and decode_center_size
  std::string code_type{"encode_center_size"};
  bool box_normalized{true};
  int axis{0};
  std::vector<float> variance{};
};

/// ----------------------- multiclass_nms operators ----------------------
struct MulticlassNmsParam : ParamBase {
  const lite::Tensor* bboxes{};
  const lite::Tensor* scores{};
  lite::Tensor* out{};
  lite::Tensor* index{};
  int background_label{0};
  float score_threshold{};
  int nms_top_k{};
  float nms_threshold{0.3f};
  float nms_eta{1.0f};
  int keep_top_k;
  bool normalized{true};
  const lite::Tensor* rois_num{};
  lite::Tensor* nms_rois_num{};
};

/// ----------------------- matrix_nms operators ----------------------
struct MatrixNmsParam : ParamBase {
  const lite::Tensor* bboxes{};
  const lite::Tensor* scores{};
  lite::Tensor* out{};
  lite::Tensor* index{};
  lite::Tensor* rois_num{};
  int background_label{0};
  float score_threshold{};
  float post_threshold{0.0f};
  int nms_top_k{};
  int keep_top_k;
  bool normalized{true};
  bool use_gaussian{false};
  float gaussian_sigma{2.0f};
};

/// ----------------------- priorbox operators ----------------------
struct PriorBoxParam : ParamBase {
  lite::Tensor* input{};
  lite::Tensor* image{};
  lite::Tensor* boxes{};
  lite::Tensor* variances{};

  bool flip{true};
  bool clip{true};
  std::vector<float> min_sizes;
  std::vector<float> max_sizes;
  std::vector<float> aspect_ratios;
  std::vector<float> variances_;
  int img_w{0};
  int img_h{0};
  float step_w{0.f};
  float step_h{0.f};
  float offset{0.5f};
  int prior_num{0};
  bool flatten_to_2d{false};
  // priortype: prior_min, prior_max, prior_com
  std::vector<std::string> order;
  bool min_max_aspect_ratios_order{false};
};

struct DensityPriorBoxParam : public PriorBoxParam {
  std::vector<float> fixed_sizes;
  std::vector<float> fixed_ratios;
  std::vector<int> density_sizes;
};
/// ----------------------- GRU operators ----------------------f
struct GRUParam : ParamBase {
  const lite::Tensor* input{nullptr};
  const lite::Tensor* h0{nullptr};
  const lite::Tensor* weight{nullptr};
  const lite::Tensor* bias{nullptr};
  lite::Tensor* batch_gate{nullptr};
  lite::Tensor* batch_reset_hidden_prev{nullptr};
  lite::Tensor* batch_hidden{nullptr};
  lite::Tensor* hidden{nullptr};

  std::string gate_activation{"sigmoid"};
  std::string activation{"tanh"};
  bool is_reverse{false};
  bool origin_mode{false};

  // for int8
  WITH_INT8_CONFIG
};

struct BiGRUParam : ParamBase {
  const lite::Tensor* input{nullptr};
  const lite::Tensor* fw_mul_w{nullptr};
  const lite::Tensor* fw_mul_b{nullptr};
  const lite::Tensor* fw_gru_w{nullptr};
  const lite::Tensor* fw_gru_b{nullptr};
  const lite::Tensor* bw_mul_w{nullptr};
  const lite::Tensor* bw_mul_b{nullptr};
  const lite::Tensor* bw_gru_w{nullptr};
  const lite::Tensor* bw_gru_b{nullptr};
  lite::Tensor* fw_output{nullptr};
  lite::Tensor* bw_output{nullptr};

  int fw_mul_x_num_col_dims{1};
  int fw_mul_y_num_col_dims{1};
  int bw_mul_x_num_col_dims{1};
  int bw_mul_y_num_col_dims{1};

  std::string fw_gru_gate_activation{"sigmoid"};
  std::string fw_gru_activation{"tanh"};
  std::string bw_gru_gate_activation{"sigmoid"};
  std::string bw_gru_activation{"tanh"};
  bool fw_gru_origin_mode{false};
  bool bw_gru_origin_mode{false};
  bool has_mul_b{false};
  bool has_gru_b{false};
};

/// ----------------------- BeamSearchDecode operators ----------------------f
struct BeamSearchDecodeParam : ParamBase {
  std::vector<lite::Tensor>* ids{nullptr};
  std::vector<lite::Tensor>* scores{nullptr};
  lite::Tensor* sentence_ids{nullptr};
  lite::Tensor* sentence_scores{nullptr};
  int beam_size;
  int end_id;
};

/// ----------------------- LookupTable operators ----------------------f
struct LookupTableParam : ParamBase {
  const lite::Tensor* W{nullptr};
  const lite::Tensor* Ids{nullptr};
  lite::Tensor* Out{nullptr};
  int64_t padding_idx{-1};
  bool is_test{true};
  std::string entry_config{""};  // used in distributed training
  std::string entry{"none"};
};

struct LookupTableDequantParam : ParamBase {
  lite::Tensor* W{nullptr};
  lite::Tensor* Ids{nullptr};
  lite::Tensor* Out{nullptr};
  int64_t padding_idx{-1};
};

struct Im2SequenceParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  std::vector<int> kernels{3, 3};
  std::vector<int> strides{1, 1};
  std::vector<int> paddings{0, 0, 0, 0};
  std::vector<int> out_strides{1, 1};
};

struct SequenceSoftmaxParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct NormParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* Norm{};
  int axis{1};
  float epsilon{1e-10f};
};
struct LayerNormParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Scale{};
  const lite::Tensor* Bias{};
  lite::Tensor* Y{};
  lite::Tensor* Mean{};
  lite::Tensor* Variance{};
  int begin_norm_axis{1};
  float epsilon{1e-5f};
};

struct LogicalParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
};

struct CompareParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  bool force_cpu{0};
  int axis{-1};
  bool fuse_greater_than{0};
  lite::Tensor* Out{};
};

struct WhileParam : ParamBase {
  Tensor* cond{};
  int block_idx{-1};
  std::shared_ptr<const cpp::ProgramDesc> program_desc{nullptr};
  Scope* exec_scope{nullptr};
};

struct TopkParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* KTensor{};
  lite::Tensor* Out{};
  lite::Tensor* Indices{};
  bool k_is_tensor{false};
  int K{1};
  int axis{-1};
};

struct IncrementParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  float step{1.f};
};

struct WriteToArrayParam : ParamBase {
  const lite::Tensor* X{nullptr};
  const lite::Tensor* I{nullptr};
  std::vector<lite::Tensor>* Out{nullptr};
};

struct ReadFromArrayParam : ParamBase {
  const std::vector<lite::Tensor>* X{nullptr};
  const lite::Tensor* I{nullptr};
  lite::Tensor* Out{nullptr};
};

struct BeamSearchParam : ParamBase {
  const lite::Tensor* pre_ids{};
  const lite::Tensor* pre_scores{};
  const lite::Tensor* ids{};
  const lite::Tensor* scores{};
  lite::Tensor* selected_ids{};
  lite::Tensor* selected_scores{};
  lite::Tensor* parent_idx{};
  int level;
  int beam_size;
  int end_id;
  bool is_accumulated;
};

struct SequencePoolParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* MaxIndex{};
  std::string pool_type{"AVERAGE"};
  float pad_value{0.0f};
};

struct SequenceConvParam : ParamBase {
  const lite::Tensor* X{};
  // not const for python unit_test
  lite::Tensor* Filter{};
  lite::Tensor* Out{};
  int contextStart{0};
  int contextStride{1};
  int contextLength;
};

struct SequencePoolConcatParam : ParamBase {
  std::vector<lite::Tensor*> X{};
  lite::Tensor* Out{};
  std::vector<std::string> pool_type{};
};

struct SequencePoolGradParam : ParamBase {
  const lite::Tensor* X{};
  std::string pool_type{"AVERAGE"};
#ifdef LITE_WITH_X86
  float pad_value{0.0f};
#endif
  // for backward
  const lite::Tensor* Out_Grad{};
  const lite::Tensor* MaxIndex_Grad{};
  lite::Tensor* X_Grad{};
};

struct SearchGroupPaddingParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* out_emb_padding{};
  lite::Tensor* out_new{};
  lite::Tensor* out_padding{};
  int pad_id;
};

struct SequenceReshapeParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* output{};
  int new_dim;
};

struct SequenceExpandParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  int ref_level{-1};
};

struct SequencePadParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* PadValue{};
  lite::Tensor* Out{};
  lite::Tensor* Length{};
  int padded_length{-1};
};

struct SequenceUnpadParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Length{};
  lite::Tensor* Out{};
};

struct SequenceMaskParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* MaxLenTensor{nullptr};
  lite::Tensor* Y{};
  int maxlen{-1};
  int out_dtype;
};

struct SequenceExpandAsParam : ParamBase {
  const lite::Tensor* x{nullptr};
  const lite::Tensor* y{nullptr};
  lite::Tensor* out{nullptr};
};

struct SequenceReverseParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct SequenceConcatParam : ParamBase {
  std::vector<lite::Tensor*> X{};
  lite::Tensor* Out{};
};

struct MeshgridParam : ParamBase {
  std::vector<lite::Tensor*> X{};
  std::vector<lite::Tensor*> Out{};
};

struct AttentionPaddingMaskParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  int pad_id;
  float mask;
  lite::Tensor* Out{};
  lite::Tensor* pad_begin{};
};

struct SequenceArithmeticParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  int op_type{1};
  lite::Tensor* Out{};
};

struct LodResetParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  std::vector<int> target_lod;
  bool append;
};

struct IsEmptyParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct ReduceParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  std::vector<int> dim{0};
  bool keep_dim{false};
  bool reduce_all{false};
};

struct VarConv2DParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* ROW{};
  const lite::Tensor* COLUMN{};
  const lite::Tensor* W{};
  lite::Tensor* Out{};
  lite::Tensor* Col{};

  int input_channel;
  int output_channel;
  int stride_h;
  int stride_w;
  int kernel_h;
  int kernel_w;

  bool fuse_relu{false};

#ifdef LITE_WITH_XPU
  bool __xpu__float_to_fix{false};  // Is W already converted to int16/int8
  float __xpu__w_max{0.0f};         // Abs max in W
#endif
};

/// ----------------------- shape operators ----------------------
struct ShapeParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct CastParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  int out_dtype{2};
  int in_dtype{2};
};

struct SliceParam : ParamBase {
  const lite::Tensor* X{nullptr};
  lite::Tensor* Out{nullptr};
  const std::vector<lite::Tensor>* XTensorList{nullptr};
  std::vector<lite::Tensor>* OutTensorList{nullptr};
  std::vector<int> axes{};
  std::vector<int> starts{};
  std::vector<int> ends{};
  std::vector<int> decrease_axis{};
  std::vector<int> infer_flags{};
  std::vector<lite::Tensor*> StartsTensorList{};
  std::vector<lite::Tensor*> EndsTensorList{};
  const lite::Tensor* StartsTensor{nullptr};
  const lite::Tensor* EndsTensor{nullptr};
};

struct AffineChannelParam : ParamBase {
  const lite::Tensor* X{};  // X is 4D tensor
  const lite::Tensor* Scale{};
  const lite::Tensor* Bias{};
  std::string data_layout{"NCHW"};  // optional string from: NHWC, NCHW.
  lite::Tensor* Out{};
};

struct AffineGridParam : ParamBase {
  const lite::Tensor* X{};  // Theta:shape {?, 2, 3}
  std::vector<int> output_shape;
  const lite::Tensor* OutputShape;
  lite::Tensor* Out{};
  bool align_corners{true};
};

struct AnchorGeneratorParam : ParamBase {
  const lite::Tensor* Input{};
  std::vector<float> anchor_sizes{};
  std::vector<float> aspect_ratios{};
  std::vector<float> stride{};
  std::vector<float> variances{{0.1f, 0.1f, 0.2f, 0.2f}};
  float offset{0.5f};

  lite::Tensor* Anchors{};
  lite::Tensor* Variances{};
};

struct GenerateProposalsParam : ParamBase {
  // inputs
  const lite::Tensor* Scores{};
  const lite::Tensor* BboxDeltas{};
  const lite::Tensor* ImInfo{};
  lite::Tensor* Anchors{};
  lite::Tensor* Variances{};

  // attrs
  int pre_nms_topN{6000};
  int post_nms_topN{1000};
  float nms_thresh{0.5f};
  float min_size{0.1f};
  float eta{1.0f};

  // outputs
  lite::Tensor* RpnRois{};
  lite::Tensor* RpnRoiProbs{};
  lite::Tensor* RpnRoisLod{};
  lite::Tensor* RpnRoisNum{};
};

struct GenerateProposalsV2Param : ParamBase {
  // inputs
  const lite::Tensor* Scores{};
  const lite::Tensor* BboxDeltas{};
  const lite::Tensor* ImShape{};
  lite::Tensor* Anchors{};
  lite::Tensor* Variances{};

  // attrs
  int pre_nms_topN{6000};
  int post_nms_topN{1000};
  float nms_thresh{0.5f};
  float min_size{0.1f};
  float eta{1.0f};
  bool pixel_offset{true};

  // outputs
  lite::Tensor* RpnRois{};
  lite::Tensor* RpnRoiProbs{};
  lite::Tensor* RpnRoisLod{};
  lite::Tensor* RpnRoisNum{};
};

/// ----------------------- squeeze operators ----------------------
struct SqueezeParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* XShape{};
  std::vector<int> axes{};
  bool inplace{false};
};

struct UnsqueezeParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* XShape{};
  std::vector<int> axes{};
  const lite::Tensor* axes_tensor{};
  std::vector<const lite::Tensor*> axes_tensor_vct{};
  bool inplace{false};
};

/// ----------------------- expand operators ----------------------
struct ExpandParam : ParamBase {
  const lite::Tensor* X{nullptr};
  const lite::Tensor* ExpandTimes{nullptr};
  std::vector<lite::Tensor*> expand_times_tensor{};
  lite::Tensor* Out{nullptr};
  std::vector<int> expand_times{};
};

/// ----------------------- expand v2 operators ----------------------
struct ExpandV2Param : ParamBase {
  const lite::Tensor* X{nullptr};
  const lite::Tensor* Shape{nullptr};
  std::vector<lite::Tensor*> expand_shapes_tensor{};
  lite::Tensor* Out{nullptr};
  std::vector<int> shape{};
};

/// ----------------------- expand as operators ----------------------
struct ExpandAsParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Target{};
  lite::Tensor* Out{};
};

/// ----------------------- matmul operators ----------------------
struct MatMulParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  bool transpose_X{false};
  bool transpose_Y{false};
  float alpha{1.0f};
  WITH_INT8_CONFIG
};

struct BmmParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
};

struct GatherNdParam : ParamBase {
  const lite::Tensor* x{nullptr};
  const lite::Tensor* index{nullptr};
  lite::Tensor* out{nullptr};
};

struct GatherParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Index{};
  const lite::Tensor* Axis{nullptr};
  lite::Tensor* Out{};
  int axis{-1};
};

struct GatherTreeParam : ParamBase {
  const lite::Tensor* ids{nullptr};
  const lite::Tensor* parents{nullptr};
  lite::Tensor* out{nullptr};
};

/// ----------------------- assign operators -----------------------
struct AssignParam : ParamBase {
  // for tensor
  const lite::Tensor* X{nullptr};
  lite::Tensor* Out{nullptr};

  // for tensor_array
  const std::vector<lite::Tensor>* X_array{nullptr};
  std::vector<lite::Tensor>* Out_array{nullptr};
};

/// ----------------------- roi_align operators -----------------------
struct RoiAlignParam : ParamBase {
  lite::Tensor* X{nullptr};
  lite::Tensor* ROIs{nullptr};
  lite::Tensor* RoisLod{nullptr};
  lite::Tensor* RoisNum{nullptr};
  lite::Tensor* Out{nullptr};
  float spatial_scale{1.0f};
  int pooled_height{1};
  int pooled_width{1};
  int sampling_ratio{-1};
  bool align{false};
};

/// ----------------------- box_clip operators -----------------------
struct BoxClipParam : ParamBase {
  const lite::Tensor* Input{};
  const lite::Tensor* ImInfo{};
  lite::Tensor* Output{};
};

struct RangeParam : ParamBase {
  const lite::Tensor* Start;
  const lite::Tensor* End;
  const lite::Tensor* Step;
  lite::Tensor* Out;
};

/// ----------------------- assign_value operators -----------------------
struct AssignValueParam : ParamBase {
  std::vector<int> shape{};
  int dtype{};
  std::vector<float> fp32_values{};
  std::vector<int> int32_values{};
  std::vector<int64_t> int64_values{};
  std::vector<int> bool_values{};
  lite::Tensor* Out{};
};

/// --------------- sequence_topk_avg_pooling operators ------------------
struct SequenceTopkAvgPoolingParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* ROW{};
  const lite::Tensor* COLUMN{};
  lite::Tensor* Out{};
  lite::Tensor* pos{};
  int channel_num{};
  std::vector<int> topks{};
};

/// --------------- topk_pooling operators ------------------
struct TopkPoolingParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  int top_k{1};
  int feat_map_num{1};
};

/// --------------- search_fc operators ------------------
struct SearchFcParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* W{};
  const lite::Tensor* b{};
  lite::Tensor* Out{};
  int out_size{};

  bool fuse_relu{false};

#ifdef LITE_WITH_XPU
  bool __xpu__float_to_fix{false};  // Is W already converted to int16/int8
  float __xpu__w_max{0.0f};         // Abs max in W
#endif
};
/// --------------------- match_matrix_tensor operators --------------------
struct MatchMatrixTensorParam : ParamBase {
  const lite::Tensor* x{};
  const lite::Tensor* y{};
  const lite::Tensor* w{};
  lite::Tensor* out{};
  lite::Tensor* tmp{};

  int dim_t;
  bool fuse_relu{false};

#ifdef LITE_WITH_XPU
  bool __xpu__float_to_fix{false};  // Is w already converted to int16/int8
  float __xpu__w_max{0.0f};         // Abs max in w
#endif
};

/// --------------------- search_seq_depadding operators --------------------
struct SearchSeqDepaddingParam : ParamBase {
  const lite::Tensor* pad{};
  const lite::Tensor* src{};
  lite::Tensor* out{};
};

/// --------------------- search_grnn operators --------------------
struct SearchGrnnParam : ParamBase {
  const lite::Tensor* x{};
  const lite::Tensor* wi{};
  const lite::Tensor* wh{};
  int num_input;
  int num_hidden;

  lite::Tensor* out{};
  lite::Tensor* tmp_buffer{};
  lite::Tensor* idx_sorted_by_width{};
  lite::Tensor* layout_input{};

#ifdef LITE_WITH_XPU
  bool __xpu__float_to_fix{false};   // Is wi/wh already converted to int16/int8
  std::vector<float> __xpu__wi_max;  // Abs max in wi
  std::vector<float> __xpu__wh_max;  // Abs max in wh
#endif
};

struct SplitLodTensorParam : ParamBase {
  const lite::Tensor* x{};
  const lite::Tensor* mask{};
  lite::Tensor* out_true{};
  lite::Tensor* out_false{};
  int level{};
};

struct MergeLodTensorParam : ParamBase {
  const lite::Tensor* x{};
  const lite::Tensor* mask{};
  const lite::Tensor* in_true{};
  const lite::Tensor* in_false{};
  lite::Tensor* out{};
  int level{};
};

struct ConditionalBlockParam : ParamBase {
  const lite::Tensor* cond{};
  std::vector<lite::Tensor*> inputs{};
  std::vector<lite::Tensor*> outs{};
  int block_idx{-1};
  std::shared_ptr<const cpp::ProgramDesc> program_desc{nullptr};
  Scope* exec_scope{nullptr};
  bool is_scalar_condition{};
};

struct CollectFpnProposalsParam : ParamBase {
  std::vector<lite::Tensor*> multi_level_rois{};
  std::vector<lite::Tensor*> multi_level_scores{};
  std::vector<lite::Tensor*> multi_rois_num{};
  lite::Tensor* rois_num{};
  lite::Tensor* fpn_rois{};
  int post_nms_topN{};
};

struct DistributeFpnProposalsParam : ParamBase {
  const lite::Tensor* fpn_rois{};
  const lite::Tensor* rois_num{};
  std::vector<lite::Tensor*> multi_fpn_rois{};
  std::vector<lite::Tensor*> multi_rois_num{};
  lite::Tensor* restore_index{};
  int min_level{};
  int max_level{};
  int refer_level{};
  int refer_scale{};
  bool pixel_offset{true};
};

/// --------------------- instance_norm operators --------------------
struct InstanceNormParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* out{};
  lite::Tensor* bias{};
  lite::Tensor* scale{};
  lite::Tensor* saved_mean{};
  lite::Tensor* saved_variance{};
  float epsilon;
  bool fuse_relu{false};
  std::string activation_type{""};
  float alpha{6.f};
};
/// --------------------- group_norm operators --------------------
struct GroupNormParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* out{};
  lite::Tensor* bias{};
  lite::Tensor* scale{};
  lite::Tensor* saved_mean{};
  lite::Tensor* saved_variance{};
  std::string data_layout_str{"NCHW"};
  float epsilon;
  int groups;
  int channels;
};

/// --------------------- grid sampler operators --------------------
struct GridSamplerParam : ParamBase {
  const lite::Tensor* x{nullptr};
  const lite::Tensor* grid{nullptr};
  lite::Tensor* out{nullptr};
  bool align_corners{true};
  std::string padding_mode{"zeros"};
  std::string mode{"bilinear"};
};

struct LstmParam : ParamBase {
  lite::Tensor* Input{};
  lite::Tensor* Weight{};
  lite::Tensor* Bias{};
  lite::Tensor* Hidden{};
  lite::Tensor* Cell{};
  lite::Tensor* BatchGate{};
  lite::Tensor* BatchCellPreAct{};
  lite::Tensor* H0{nullptr};
  lite::Tensor* C0{nullptr};
  bool use_peepholes;
  bool is_reverse;
  lite_api::ActivationType gate_activation;
  lite_api::ActivationType cell_activation;
  lite_api::ActivationType candidate_activation;
  // for int8
  WITH_INT8_CONFIG
};

struct CrfDecodingParam : ParamBase {
  lite::Tensor* emission{};
  lite::Tensor* transition{};
  lite::Tensor* label{};
  lite::Tensor* length{};
  lite::Tensor* viterbi_path{};
};

struct CtcAlignParam : ParamBase {
  lite::Tensor* input{};
  lite::Tensor* input_length{};
  lite::Tensor* output{};
  lite::Tensor* output_length{};
  int blank{0};
  bool merge_repeated{true};
  int padding_value{0};
};

struct XPUResNet50Param : ParamBase {
  lite::Tensor* input{};
  std::vector<lite::Tensor*> filter;
  std::vector<lite::Tensor*> bias;
  std::vector<lite::Tensor*> max_filter;
  lite::Tensor* output{};
};

struct XPUSoftmaxTopkParam : ParamBase {
  const lite::Tensor* x{};
  lite::Tensor* output{};
  lite::Tensor* indices{};
  int axis{-1};
  int K{1};
};

struct XPUMultiSoftmaxParam : ParamBase {
  const lite::Tensor* input{};
  lite::Tensor* concat_output{};
  std::vector<lite::Tensor*> output;
  std::vector<int> lod;
};

struct XPUBlockFuseParam : ParamBase {
  const lite::Tensor* input{nullptr};
  const lite::Tensor* filter{nullptr};
  const lite::Tensor* bias{nullptr};
  const lite::Tensor* branch{nullptr};
  const lite::Tensor* input_max{nullptr};
  lite::Tensor* output{nullptr};
  lite::Tensor* output_max{nullptr};
  std::vector<int> op_type;
  std::vector<int> place_x;
  std::vector<int> place_y;
  std::vector<int> place_z;
  std::vector<int> filter_dims;
  std::vector<int> strides;
  std::shared_ptr<std::vector<int>> paddings;
  std::shared_ptr<std::vector<int>> dilations;
  std::vector<int> groups;
  std::vector<int> act_type;
  std::vector<float> act_param;
  std::vector<int> conv_bias;
  std::vector<int> block_lod;
  bool has_bias{false};
  bool has_branch{false};
  // for int8/int16
  bool enable_int8{false};
  bool enable_int16{false};
  float quant_input_max{0.f};
  float quant_w_max{0.f};
  float quant_output_max{0.f};
  float quant_branch_max{0.f};
};

struct XPUMultiEncoderParam : ParamBase {
  lite::Tensor* input{};
  std::vector<lite::Tensor*> fc_weight;
  std::vector<lite::Tensor*> fc_bias;
  std::vector<lite::Tensor*> ln_scale;
  std::vector<lite::Tensor*> ln_bias;
  const lite::Tensor* mask{nullptr};
  const lite::Tensor* SeqLod{nullptr};
  const lite::Tensor* PadSeqLen{nullptr};
  lite::Tensor* output{nullptr};

  std::vector<int> slice_axes{};
  std::vector<int> slice_starts{};
  std::vector<int> slice_ends{};
  std::vector<int> slice_decrease_axis{};
  std::vector<float> input_max{};
  std::vector<lite::Tensor*> weight_max{};
  std::vector<std::string> quant_types{};
  int n_layers{};
  int head_num{};
  int size_per_head{};
  int hidden_dim{};
  std::string act_type{};
  std::string precision{};
  bool enable_qkv_fusion{false};
  bool norm_before{false};
  bool adaptive_seqlen{false};
  bool per_channel{false};
};

struct XPUEmbeddingWithEltwiseAddParam : ParamBase {
  std::vector<lite::Tensor*> Ids;
  std::vector<lite::Tensor*> Tables;
  const lite::Tensor* Mask{nullptr};
  lite::Tensor* SeqLod{nullptr};
  lite::Tensor* PadSeqLen{nullptr};
  lite::Tensor* Out{nullptr};
  int64_t padding_idx{-1};
};

struct XPUFcParam : ParamBase {
  const lite::Tensor* input{nullptr};
  const lite::Tensor* w{nullptr};
  const lite::Tensor* bias{nullptr};
  const lite::Tensor* input_max{nullptr};
  lite::Tensor* output{nullptr};
  lite::Tensor* output_max{nullptr};
  lite::DDim in_mat_dims;

  int act_type;
  float act_param;
  std::vector<float> weight_max{};
  std::string precision{};
  bool has_bias{false};
  int in_num_col_dims{1};
  bool transpose_x{false};
  bool transpose_w{true};

  // int8/int16
  bool enable_int8{false};
  bool enable_int16{false};
  float quant_input_max{0.f};
  float quant_output_max{0.f};
  bool per_channel{false};
};

struct XPUResNetCbamParam : ParamBase {
  lite::Tensor* input{};
  std::vector<lite::Tensor*> filter;
  std::vector<lite::Tensor*> bias;
  std::vector<lite::Tensor*> max_filter;
  lite::Tensor* output{};

  float pool_p{1.0f};
};

struct XPUMmdnnSearchAttentionParam : ParamBase {
  lite::Tensor* X{};
  lite::Tensor* W{};
  lite::Tensor* b{};
  lite::Tensor* Out{};

  float W_max{0.0f};
  int pad_id{0};
  float alpha0{1.0f};
  float alpha1{1.0f};
  float mask{1.0f};
};

struct XPUMmdnnBidEmbGrnnAttParam : ParamBase {
  lite::Tensor* id0{};
  lite::Tensor* id1{};
  lite::Tensor* emb_tbl{};
  lite::Tensor* grnn_fw_wh{};
  lite::Tensor* grnn_fw_wi{};
  lite::Tensor* grnn_rv_wh{};
  lite::Tensor* grnn_rv_wi{};
  lite::Tensor* att_fc_w{};
  lite::Tensor* att_fc_b{};

  std::vector<float> grnn_fw_wh_maxs;
  std::vector<float> grnn_fw_wi_maxs;
  std::vector<float> grnn_rv_wh_maxs;
  std::vector<float> grnn_rv_wi_maxs;
  float att_fc_w_max{0.0f};

  lite::Tensor* grnn_fw_pool_out{};
  lite::Tensor* grnn_rv_pool_out{};
  lite::Tensor* att_pool_out{};
  lite::Tensor* concat_3in1_out{};
  lite::Tensor* emb_fw_out{};
};

struct XPUMmdnnBidEmbGrnnAttParam2 : ParamBase {
  lite::Tensor* id0{};
  lite::Tensor* id1{};
  lite::Tensor* emb_tbl{};
  lite::Tensor* grnn_fw_wh{};
  lite::Tensor* grnn_fw_wi{};
  lite::Tensor* grnn_rv_wh{};
  lite::Tensor* grnn_rv_wi{};
  lite::Tensor* att_fc_w{};
  lite::Tensor* att_fc_b{};

  std::vector<float> grnn_fw_wh_maxs;
  std::vector<float> grnn_fw_wi_maxs;
  std::vector<float> grnn_rv_wh_maxs;
  std::vector<float> grnn_rv_wi_maxs;
  float att_fc_w_max{0.0f};

  lite::Tensor* emb0_out{};
  lite::Tensor* grnn_fw_pool_out{};
  lite::Tensor* grnn_rv_pool_out{};
  lite::Tensor* att_pool_out{};
  lite::Tensor* concat_3in1_out{};
  lite::Tensor* emb_fw_out{};
};

struct XPUMmdnnBidEmbAttParam : ParamBase {
  lite::Tensor* id0{};
  lite::Tensor* id1{};
  lite::Tensor* emb_tbl{};
  lite::Tensor* att_fc_w{};
  lite::Tensor* att_fc_b{};

  float att_fc_w_max{0.0f};

  lite::Tensor* att_pool_out{};
  lite::Tensor* emb_fw_out{};
};

struct XPUMmdnnMatchConvTopkParam : ParamBase {
  lite::Tensor* input_x{};
  lite::Tensor* input_y{};
  lite::Tensor* input_w{};
  lite::Tensor* conv_w{};

  float input_w_max{0.0f};
  float conv_w_max{0.0f};
  std::vector<int> topks;
  int output_channel{0};
  int channel_num{0};
  int dim_t{0};

  lite::Tensor* topk_out{};
};

struct XPUMmdnnMergeAllParam : ParamBase {
  std::vector<lite::Tensor*> concat_7in1_x;
  std::vector<lite::Tensor*> concat_topk_x;
  lite::Tensor* grnn_fw_wh{};
  lite::Tensor* grnn_fw_wi{};
  lite::Tensor* grnn_rv_wh{};
  lite::Tensor* grnn_rv_wi{};
  lite::Tensor* fc0_w{};
  lite::Tensor* fc0_b{};
  lite::Tensor* fc1_w{};
  lite::Tensor* fc1_b{};
  lite::Tensor* fc2_w{};
  lite::Tensor* fc2_b{};

  std::vector<float> grnn_fw_wh_maxs;
  std::vector<float> grnn_fw_wi_maxs;
  std::vector<float> grnn_rv_wh_maxs;
  std::vector<float> grnn_rv_wi_maxs;
  float fc0_w_max{0.0f};
  float fc1_w_max{0.0f};
  float fc2_w_max{0.0f};

  lite::Tensor* out{};
};

struct XPUSfaHeadParam : ParamBase {
  lite::Tensor* input{nullptr};
  lite::Tensor* output{nullptr};

  std::string op_type{""};
};

struct XPUGenerateSequenceParam : ParamBase {
  const lite::Tensor* input{nullptr};
  lite::Tensor* output{nullptr};

  int axis{-1};
  bool flatten{false};
  float value{0.f};
  int dtype{-1};
};

struct XPULogitParam : ParamBase {
  const lite::Tensor* input{nullptr};
  lite::Tensor* output{nullptr};
  float eps{1e-7f};
};

struct XPUConvPixelShuffleFuseParam : ParamBase {
  const lite::Tensor* input{nullptr};
  const lite::Tensor* filter_0{nullptr};
  const lite::Tensor* filter_1{nullptr};
  const lite::Tensor* bias_0{nullptr};
  const lite::Tensor* bias_1{nullptr};
  const lite::Tensor* input_max{nullptr};
  lite::Tensor* output{nullptr};
  lite::Tensor* output_max{nullptr};
  std::vector<int> strides_0;
  std::vector<int> strides_1;
  std::shared_ptr<std::vector<int>> paddings_0;
  std::shared_ptr<std::vector<int>> paddings_1;
  std::shared_ptr<std::vector<int>> dilations_0;
  std::shared_ptr<std::vector<int>> dilations_1;
  std::vector<int> groups_0;
  std::vector<int> groups_1;
  std::vector<int> act_type_0;
  std::vector<int> act_type_1;
  std::vector<float> act_param_0;
  std::vector<float> act_param_1;
  int upscale_factor{1};
  bool has_bias_0{false};
  bool has_bias_1{false};
};

struct XPUDynamicLstmFuseParam : ParamBase {
  const lite::Tensor* input{nullptr};
  const lite::Tensor* weight_0{nullptr};
  const lite::Tensor* weight_1{nullptr};
  const lite::Tensor* bias_0{nullptr};
  const lite::Tensor* bias_1{nullptr};
  const lite::Tensor* h0{nullptr};
  const lite::Tensor* c0{nullptr};
  lite::Tensor* hidden{nullptr};
  bool has_h0{false};
  bool is_reverse{false};
};

// For DeformableConvolution op
struct DeformableConvParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* offset{};
  lite::Tensor* mask{};
  lite::Tensor* output{};
  int deformable_groups{1};
  int im2col_step{1};
  bool modulated{true};  // True-v2 False-v1
  std::string data_format{"Anylayout"};
  // convolution parameter
  ConvParam conv_param;
  // support var_length or not
  bool var_length{false};
  // only used in conv_transpose.
  std::vector<int> output_size;
};

struct PixelShuffleParam : ParamBase {
  lite::Tensor* x{nullptr};
  lite::Tensor* output{nullptr};
  int upscale_factor{1};
};

struct RetinanetDetectionOutputParam : ParamBase {
  std::vector<Tensor*> bboxes{};
  std::vector<Tensor*> scores{};
  std::vector<Tensor*> anchors{};
  Tensor* im_info{};
  Tensor* out{};
  float score_threshold{};
  int nms_top_k{};
  float nms_threshold{};
  float nms_eta{};
  int keep_top_k{};
};

struct WhereIndexParam : ParamBase {
  const lite::Tensor* input{nullptr};
  lite::Tensor* output{nullptr};
};

struct WhereParam : ParamBase {
  const lite::Tensor* x{nullptr};
  const lite::Tensor* y{nullptr};
  const lite::Tensor* condition{nullptr};
  lite::Tensor* out{nullptr};
};

struct ClipParam : ParamBase {
  Tensor* x{};
  Tensor* min_tensor{};
  Tensor* max_tensor{};
  Tensor* out{};
  float min{};
  float max{};
};

struct PrintParam : ParamBase {
  const lite::Tensor* in{};
  lite::Tensor* out{};
  std::string name;
  int first_n{-1};
  std::string message;
  int summarize{20};
  bool print_tensor_name{true};
  bool print_tensor_type{true};
  bool print_tensor_shape{true};
  bool print_tensor_lod{true};
  bool print_tensor_layout{true};
  std::string print_phase;
  bool is_forward{true};
};

struct OneHotParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* depth_tensor{nullptr};
  lite::Tensor* Out{};
  int depth;
  int dtype;
  bool allow_out_of_range;
};

struct TrigonometricParam : ParamBase {
  lite::Tensor* X{};
  lite::Tensor* Out{};
};

using SinParam = TrigonometricParam;
using CosParam = TrigonometricParam;
using TanParam = TrigonometricParam;
using AsinParam = TrigonometricParam;
using AcosParam = TrigonometricParam;
using AtanParam = TrigonometricParam;

struct FlattenContiguousRangeParam : ParamBase {
  const lite::Tensor* x{nullptr};
  lite::Tensor* out{nullptr};
  lite::Tensor* xshape{nullptr};
  int start_axis{1};
  int stop_axis{1};
};

struct LoDArrayLengthParam : ParamBase {
  std::vector<lite::Tensor>* x{};
  lite::Tensor* out{};
};

struct SelectInputParam : ParamBase {
  std::vector<lite::Tensor*> X{};
  lite::Tensor* Mask{};
  lite::Tensor* Out{};
};

struct TensorArrayToTensorParam : ParamBase {
  std::vector<lite::Tensor>* X{};
  lite::Tensor* Out{};
  lite::Tensor* OutIndex{};
  int axis{0};
  bool use_stack{false};
};

struct RnnParam : ParamBase {
  lite::Tensor* Input;
  std::vector<lite::Tensor*> PreState;
  std::vector<lite::Tensor*> WeightList;
  const lite::Tensor* SequenceLength{nullptr};
  lite::Tensor* DropoutState;
  lite::Tensor* Reserve;
  lite::Tensor* Out;
  std::vector<lite::Tensor*> State;
  float dropout_prob{0.0f};
  bool is_bidirec{false};
  int input_size{10};
  int hidden_size{100};
  int num_layers{1};
  std::string mode{"LSTM"};
  bool is_test{false};
  int seed{0};
};

struct StridedSliceParam : ParamBase {
  lite::Tensor* Input{};
  lite::Tensor* Out{};
  std::vector<int> starts{};
  std::vector<int> ends{};
  std::vector<int> strides{};
  std::vector<int> axes{};
  std::vector<int> infer_flags{};
  std::vector<int> decrease_axis{};
  std::vector<lite::Tensor*> StartsTensorList{};
  std::vector<lite::Tensor*> EndsTensorList{};
  std::vector<lite::Tensor*> StridesTensorList{};
  bool tensor_input{false};
  lite::Tensor* EndsTensor{nullptr};
  lite::Tensor* StartsTensor{nullptr};
  lite::Tensor* StridesTensor{nullptr};
};

struct TileParam : ParamBase {
  lite::Tensor* X{};
  lite::Tensor* Out{};
  std::vector<int> repeat_times{};
  lite::Tensor* RepeatTimes{};
  std::vector<lite::Tensor*> repeat_times_tensor{};
};

struct ScatterNdAddParam : ParamBase {
  const lite::Tensor* x{};
  lite::Tensor* indexs{};
  lite::Tensor* updates{};
  lite::Tensor* output{};
};

struct CumsumParam : ParamBase {
  const lite::Tensor* X{nullptr};
  lite::Tensor* Out{nullptr};

  int axis{-1};
  bool flatten{false};
  bool exclusive{false};
  bool reverse{false};
};

struct SamplingIdParam : ParamBase {
  const lite::Tensor* x{};
  lite::Tensor* out{};

  float min{0.f};
  float max{1.f};
  int seed{0};
};

struct PolygonBoxTransformParam : ParamBase {
  const lite::Tensor* input{nullptr};
  lite::Tensor* output{nullptr};
};

struct SumParam : ParamBase {
  std::vector<lite::Tensor*> X{};
  lite::Tensor* Out{};
  int inplace{0};
};

struct PNormParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};

  float porder{2.f};
  int axis{-1};
  float epsilon{1.0e-12f};
  bool keepdim{false};
  bool asvector{false};
};

struct LinspaceParam : ParamBase {
  const lite::Tensor* Start{};
  const lite::Tensor* Stop{};
  const lite::Tensor* Num{};
  lite::Tensor* Out{};
  int dtype{};
};

struct RoiPerspectiveTransformParam : ParamBase {
  const lite::Tensor* x{nullptr};
  const lite::Tensor* rois{nullptr};
  lite::Tensor* out{nullptr};
  lite::Tensor* mask{nullptr};
  lite::Tensor* transfor_matrix{nullptr};
  lite::Tensor* out2in_idx{nullptr};
  lite::Tensor* out2in_weight{nullptr};
  float spatial_scale{1.f};
  int transformed_height{1};
  int transformed_width{1};
};

struct CorrelationParam : ParamBase {
  const lite::Tensor* input1{nullptr};
  const lite::Tensor* input2{nullptr};
  lite::Tensor* output{nullptr};
  int pad_size;
  int kernel_size;
  int max_displacement;
  int stride1;
  int stride2;
  int corr_type_multiply{1};
};

struct ArgsortParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* Indices{};

  int axis{-1};
  bool descending{false};
};

struct FlipParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};

  std::vector<int> axis;
};

struct CosSimParam : ParamBase {
  const lite::Tensor* x{nullptr};
  const lite::Tensor* y{nullptr};
  lite::Tensor* out{nullptr};
  lite::Tensor* x_norm{nullptr};
  lite::Tensor* y_norm{nullptr};
};

struct WriteBackParam : ParamBase {
  bool tensor_array_copy{false};
  const lite::Tensor* x{};
  lite::Tensor* y{};

  std::vector<lite::Tensor>* array_x{};
  std::vector<lite::Tensor>* array_y{};
};

struct UniqueWithCountsParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* Index{};
  lite::Tensor* Count{};
};

struct GaussRandomParam : ParamBase {
  const lite::Tensor* ShapeTensor{nullptr};
  std::vector<const lite::Tensor*> ShapeTensorList{};
  std::vector<int64_t> shape{};
  lite::Tensor* Out{};
  int seed{0};
  int dtype{5};
  float mean{0.f};
  float gauss_std{0.f};
};

struct RollParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* ShiftsTensor{};
  lite::Tensor* Out{};
  std::vector<int64_t> shifts{};
  std::vector<int64_t> axis{};
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
