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
#include <string>
#include <vector>
#include "lite/api/paddle_place.h"
#include "lite/core/scope.h"
#include "lite/core/tensor.h"
#include "lite/core/types.h"
#include "lite/model_parser/cpp/block_desc.h"
#include "lite/model_parser/desc_apis.h"
#include "lite/utils/all.h"
/*
 * This file contains all the argument parameter data structure for operators.
 */

namespace paddle {
namespace lite {
namespace operators {

using param_t = Any;
#define WITH_INT8_CONFIG             \
  bool enable_int8{false};           \
  float input_scale{1.0};            \
  std::vector<float> weight_scale{}; \
  float output_scale{1.0};

/// ----------------------- Functional operators ------------------------------
struct FeedParam {
  std::vector<lite::Tensor>* feed_list{};
  lite::Tensor* out{};
  int col;
};

struct FetchParam {
  const lite::Tensor* input{};
  std::vector<lite::Tensor>* fetch_list{};
  int col;
};

// Helper op for lite framework
struct IoCopyParam {
  const lite::Tensor* x{};
  lite::Tensor* y{};
};

struct LayoutParam {
  const lite::Tensor* x{};
  lite::Tensor* y{};
};

struct CalibParam {
  const lite::Tensor* input{};
  lite::Tensor* output{};
  float scale;
};

struct GraphParam {
  std::vector<const lite::Tensor*> inputs{};
  std::vector<lite::Tensor*> outputs{};
  std::string model_name{"model"};
};

/// -------------------------- NN operators ------------------------------------

struct FcParam {
  lite::Tensor* input{nullptr};
  lite::Tensor* w{nullptr};
  lite::Tensor* bias{nullptr};
  lite::Tensor* output{nullptr};
  lite::DDim in_mat_dims;
  int in_num_col_dims{1};
  bool weight_transposed{false};
  // for int8
  WITH_INT8_CONFIG
};

// For Interpolate Op
struct InterpolateParam {
  lite::Tensor* X{};
  lite::Tensor* OutSize{};
  lite::Tensor* Out{};

  float scale{0.f};
  int out_h{-1};
  int out_w{-1};
  bool align_corners{true};
  std::string interp_method{"Nearest"};
};

// For Mul Op
struct MulParam {
  const lite::Tensor* x{};
  const lite::Tensor* y{};
  lite::Tensor* output{};

  int x_num_col_dims{1};
  int y_num_col_dims{1};
  // for int8
  WITH_INT8_CONFIG
};

struct MulGradParam {
  const lite::Tensor* x{};
  const lite::Tensor* y{};
  const lite::Tensor* output_grad{};
  lite::Tensor* x_grad{};
  lite::Tensor* y_grad{};

  int x_num_col_dims{1};
  int y_num_col_dims{1};
};

// For ReduceMean Op
struct ReduceMeanParam {
  lite::Tensor* X{};
  lite::Tensor* Out{};

  std::vector<int> dim;
  bool keep_dim{false};
};

// For Stack Op
struct StackParam {
  std::vector<lite::Tensor*> X;
  lite::Tensor* Out{};

  int axis{0};
};

// For Power Op
struct PowerParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};

  float scale{};
  float shift{};
  float power{};
};

struct ShuffleChannelParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};

  int group;
};

// For Yolobox
struct YoloBoxParam {
  lite::Tensor* X{};
  lite::Tensor* ImgSize{};
  lite::Tensor* Boxes{};
  lite::Tensor* Scores{};

  std::vector<int> anchors{};
  int class_num{0};
  float conf_thresh{0.f};
  int downsample_ratio{0};
};

// For Scale Op
struct ScaleParam {
  lite::Tensor* x{};
  lite::Tensor* output{};

  float scale{1.};
  float bias{};
  bool bias_after_scale{true};
};

// For Softmax op
struct SoftmaxParam {
  lite::Tensor* x{};
  lite::Tensor* output{};
  int axis{-1};
};

// For Reshape and Reshape2 Op
struct ReshapeParam {
  const lite::Tensor* x{};
  const lite::Tensor* actual_shape{nullptr};
  lite::Tensor* output{};
  lite::Tensor* xshape{};

  std::vector<int> shape{};
  bool inplace{false};
};

// For Concat op
struct ConcatParam {
  std::vector<lite::Tensor*> x{};
  lite::Tensor* output{};
  int axis{0};
};

/// ----------------------- activation operators ----------------------
struct ActivationParam {
  const lite::Tensor* X{};
  float Leaky_relu_alpha{0};   // leaky_relu param
  float Relu_clipped_coef{6};  // relu_clipped param
  std::string Prelu_mode{
      "channel"};  // prelu param, can be "all", "channel" or "element"
  lite::Tensor* Prelu_alpha{};  // prelu param
  float Swish_beta;             // swish param
  lite::Tensor* Out{};
  bool has_active{false};
  lite_api::ActivationType active_type;
};

struct ActivationGradParam {
  const lite::Tensor* X{};
  const lite::Tensor* Out{};
  // for backward
  lite::Tensor* X_grad{};
  const lite::Tensor* Out_grad{};
};

// For Convolution op
struct ConvParam {
  lite::Tensor* x{};
  lite::Tensor* filter{};
  lite::Tensor* bias{nullptr};
  lite::Tensor* residualData{nullptr};
  lite::Tensor* output{};
  std::vector<int> strides{1, 1};
  std::vector<int> paddings{0, 0};
  int groups{1};
  std::vector<int> dilations{1, 1};
  bool fuse_relu_before_depthwise_conv{false};
  bool use_mkldnn{false};
  bool fuse_relu{false};  // only used in mkldnn kernel
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
  // for int8
  WITH_INT8_CONFIG
};

// For BatchNorm op
struct BatchNormParam {
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
struct PoolParam {
  lite::Tensor* x{};
  lite::Tensor* output{};
  std::string pooling_type{""};
  std::vector<int> ksize{};
  bool global_pooling{
      false};  // if true, knernel size and paddings will be ignored
  std::vector<int> strides{1, 1};
  std::vector<int> paddings{0, 0};
  bool exclusive{true};
  bool adaptive{false};
  bool ceil_mode{false};
  bool use_quantizer{false};
  std::string data_format{"AnyLayout"};
};

// For Dropout op
struct DropoutParam {
  const lite::Tensor* x{};
  lite::Tensor* output{};
  lite::Tensor* mask{};
  float dropout_prob{.5f};
  bool is_test{false};
  bool fix_seed{false};
  int seed{0};
  std::string dropout_implementation{"downgrade_in_infer"};
};

// For Split op
struct SplitParam {
  lite::Tensor* x{};
  std::vector<lite::Tensor*> output{};
  int axis{-1};
  int num{0};
  std::vector<int> sections;
};

// For Transpose op
struct TransposeParam {
  const lite::Tensor* x{};
  lite::Tensor* output{};
  std::vector<int> axis;
  bool use_mkldnn{false};
  std::string data_format{"AnyLayout"};
};

/// ----------------------- element wise operators ----------------------
struct ElementwiseParam {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  int axis{-1};  // for broadcasting.
};

struct ElementwiseGradParam {
  const lite::Tensor* Y{};
  const lite::Tensor* Out_grad{};
  lite::Tensor* X_grad{};
  lite::Tensor* Y_grad{};
  int axis{-1};  // for broadcasting.
};

struct FusionElementwiseActivationParam : public ElementwiseParam {
  std::string act_type;
};

struct FusionElementwiseActivationGradParam : public ElementwiseGradParam {
  std::string act_type;
};

/// ----------------------- mean operators ----------------------
struct MeanParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct MeanGradParam {
  const lite::Tensor* X{};
  const lite::Tensor* Out_grad{};
  // for backward
  lite::Tensor* X_grad{};
};

/// ----------------------- fill_constant operators ----------------------
struct FillConstantParam {
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
  std::vector<int64_t> shape{};
  float value{0.0f};
  // useless for x86, keep it for compatibility
  bool force_cpu{false};
  lite::Tensor* Out{};
};

//
struct FakeQuantizeMovingAvgMaxAbsParam {
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
  float moving_rate{0.9};
};

struct FakeDequantizeMaxAbsParam {
  const lite::Tensor* x{};
  const lite::Tensor* in_scale{};
  lite::Tensor* out{};
  float max_range;
};

/// ----------------------- sgd operators ----------------------
struct SGDParam {
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};

  const lite::Tensor* Param{};
  const lite::Tensor* LearningRate{};
  const lite::Tensor* Grad{};
  lite::Tensor* ParamOut{};
};

/// ----------------------- uniform_random operators ----------------------
struct UniformRandomParam {
  std::vector<int64_t> shape{};
  float min{-1.0f};
  float max{1.0f};
  int seed{0};
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
  lite::Tensor* Out{};
};
/// ----------------------- negative operators --------------
struct NegativeParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};
/// ----------------------- pad2d operators ----------------------
struct Pad2dParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  std::vector<int> paddings{0, 0, 0, 0};
  std::string mode{"constant"};
  float pad_value = 0.f;
  std::string data_format{"NCHW"};
};

/// ----------------------- Crop operators ----------------------
struct CropParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  std::vector<int> offsets;
  std::vector<int> shape;
};

///----------------------- argmax operators ----------------------
struct ArgmaxParam {
  lite::Tensor* X{};
  lite::Tensor* Out{};
  int Axis{0};
};

///----------------------- axpy operators ----------------------
struct AxpyParam {
  lite::Tensor* Scale{};
  lite::Tensor* X{};
  lite::Tensor* Bias{};
  lite::Tensor* Out{};
};
/// ----------------------- GRU unit operators ----------------------f
struct GRUUnitParam {
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
struct LrnParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  int local_size{5};
  float alpha{1.};
  float beta{0.75};
  float k{1.};
  std::string norm_region{"AcrossChannels"};
};

/// ----------------------- decode_bboxes operators ----------------------
struct DecodeBboxesParam {
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
struct BoxCoderParam {
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
struct MulticlassNmsParam {
  const lite::Tensor* bboxes{};
  const lite::Tensor* scores{};
  lite::Tensor* out{};
  int background_label{0};
  float score_threshold{};
  int nms_top_k{};
  float nms_threshold{0.3};
  float nms_eta{1.0};
  int keep_top_k;
  bool normalized{true};
};

/// ----------------------- priorbox operators ----------------------
struct PriorBoxParam {
  lite::Tensor* input{};
  lite::Tensor* image{};
  lite::Tensor* boxes{};
  lite::Tensor* variances{};

  bool flip;
  bool clip;
  std::vector<float> min_sizes;
  std::vector<float> max_sizes;
  std::vector<float> aspect_ratios;
  std::vector<float> variances_;
  int img_w{0};
  int img_h{0};
  float step_w{0};
  float step_h{0};
  float offset{0.5};
  int prior_num{0};
  // priortype: prior_min, prior_max, prior_com
  std::vector<std::string> order;
};

struct DensityPriorBoxParam : public PriorBoxParam {
  std::vector<float> fixed_sizes;
  std::vector<float> fixed_ratios;
  std::vector<int> density_sizes;
};
/// ----------------------- GRU operators ----------------------f
struct GRUParam {
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
};

/// ----------------------- BeamSearchDecode operators ----------------------f
struct BeamSearchDecodeParam {
  std::vector<lite::Tensor>* ids{nullptr};
  std::vector<lite::Tensor>* scores{nullptr};
  lite::Tensor* sentence_ids{nullptr};
  lite::Tensor* sentence_scores{nullptr};
  int beam_size;
  int end_id;
};

/// ----------------------- LookupTable operators ----------------------f
struct LookupTableParam {
  lite::Tensor* W{nullptr};
  lite::Tensor* Ids{nullptr};
  lite::Tensor* Out{nullptr};
  int64_t padding_idx{-1};
};

struct Im2SequenceParam {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  std::vector<int> kernels{3, 3};
  std::vector<int> strides{1, 1};
  std::vector<int> paddings{0, 0, 0, 0};
  std::vector<int> out_strides{1, 1};
};

struct SequenceSoftmaxParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct NormParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  int axis{1};
  float epsilon{1e-10};
};

struct LogicalParam {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
};

struct CompareParam {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  bool force_cpu{0};
  int axis{-1};
  lite::Tensor* Out{};
};

struct WhileParam {
  Scope* scope{};
  Tensor* cond{};
  cpp::BlockDesc* sub_block{};
  std::vector<Tensor*> x{};
  std::vector<Tensor*> outs{};
};

struct TopkParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* Indices{};
  int K{1};
};

struct IncrementParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  float step{1};
};

struct WriteToArrayParam {
  const lite::Tensor* X{};
  const lite::Tensor* I{};
  std::vector<lite::Tensor>* Out{};
};

struct ReadFromArrayParam {
  std::vector<lite::Tensor>* X{};
  lite::Tensor* I{};
  lite::Tensor* Out{};
};

struct BeamSearchParam {
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

struct SequencePoolParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  std::string pool_type;
};

struct SequenceExpandParam {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  int ref_level{-1};
};

struct ReduceMaxParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  std::vector<int> dim{};
  bool keep_dim{false};
};

struct LodResetParam {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  std::vector<int> target_lod;
  bool append;
};

struct IsEmptyParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};
/// ----------------------- shape operators ----------------------
struct ShapeParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct CastParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  int out_dtype{2};
  int in_dtype{2};
};

struct SliceParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  std::vector<int> axes{};
  std::vector<int> starts{};
  std::vector<int> ends{};
  std::vector<int> decrease_axis{};
};

struct AffineChannelParam {
  const lite::Tensor* X{};  // X is 4D tensor
  const lite::Tensor* Scale{};
  const lite::Tensor* Bias{};
  std::string data_layout{"NCHW"};  // optional string from: NHWC, NCHW.
  lite::Tensor* Out{};
};

struct AnchorGeneratorParam {
  const lite::Tensor* Input{};
  std::vector<float> anchor_sizes{};
  std::vector<float> aspect_ratios{};
  std::vector<float> stride{};
  std::vector<float> variances{{0.1, 0.1, 0.2, 0.2}};
  float offset{0.5};

  lite::Tensor* Anchors{};
  lite::Tensor* Variances{};
};

struct GenerateProposalsParam {
  // inputs
  const lite::Tensor* Scores{};
  const lite::Tensor* BboxDeltas{};
  const lite::Tensor* ImInfo{};
  lite::Tensor* Anchors{};
  lite::Tensor* Variances{};

  // attrs
  int pre_nms_topN{6000};
  int post_nms_topN{1000};
  float nms_thresh{0.5};
  float min_size{0.1};
  float eta{1.0};

  // outputs
  lite::Tensor* RpnRois{};
  lite::Tensor* RpnRoiProbs{};
};
/// ----------------------- shape operators ----------------------
/// ----------------------- squeeze operators ----------------------
struct SqueezeParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* XShape{};
  std::vector<int> axes{};
};

/// ----------------------- expand operators ----------------------
struct ExpandParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  std::vector<int> expand_times{};
};

/// ----------------------- matmul operators ----------------------
struct MatMulParam {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  bool transpose_X{false};
  bool transpose_Y{false};
  float alpha{1.0f};
};

/// ----------------------- assign operators -----------------------
struct AssignParam {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
};

struct RoiAlignParam {
  lite::Tensor* X{};
  lite::Tensor* ROIs{};
  lite::Tensor* Out{};
  float spatial_scale{1.0};
  int pooled_height{1};
  int pooled_width{1};
  int sampling_ratio{-1};
};

struct BoxClipParam {
  const lite::Tensor* Input{};
  const lite::Tensor* ImInfo{};
  lite::Tensor* Output{};
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
