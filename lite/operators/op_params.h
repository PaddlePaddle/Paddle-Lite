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
#include "lite/core/scope.h"
#include "lite/core/tensor.h"
#include "lite/core/types.h"
#include "lite/model_parser/cpp/block_desc.h"
#include "lite/model_parser/desc_apis.h"
#include "lite/utils/all.h"
#include "lite/utils/variant.h"
/*
 * This file contains all the argument parameter data structure for operators.
 */

namespace paddle {
namespace lite {
namespace operators {

struct ParamBase {
 public:
  virtual ~ParamBase() {}
  virtual const std::vector<const Tensor*>* input_tensor_ptrs() {
    return nullptr;
  }
  virtual std::vector<Tensor*>* output_tensor_ptrs() { return nullptr; }

 protected:
  std::shared_ptr<std::vector<const Tensor*>> input_tensor_ptrs_cache_{nullptr};
  std::shared_ptr<std::vector<Tensor*>> output_tensor_ptrs_cache_{nullptr};
};

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
  const lite::Tensor* x{};
  lite::Tensor* y{};
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

struct SubgraphParam : ParamBase {
  std::vector<std::string> input_names{};
  std::vector<std::string> output_names{};
  std::vector<std::string> input_data_names{};
  std::vector<std::string> output_data_names{};
  int sub_block_idx{-1};
  cpp::BlockDesc* sub_block_desc{nullptr};
  Scope* scope{nullptr};
};

/// -------------------------- NN operators ------------------------------------

struct FcParam : ParamBase {
  lite::Tensor* input{nullptr};
  lite::Tensor* w{nullptr};
  lite::Tensor* bias{nullptr};
  lite::Tensor* output{nullptr};
  lite::DDim in_mat_dims;
  int in_num_col_dims{1};
  std::string activation_type{""};
  bool padding_weights{false};
  // for int8
  WITH_INT8_CONFIG
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({input}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
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
  int out_h{-1};
  int out_w{-1};
  bool align_corners{true};
  int align_mode{1};
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
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({x, y}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
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

// For ReduceMean Op
struct ReduceMeanParam : ParamBase {
  lite::Tensor* X{};
  lite::Tensor* Out{};

  std::vector<int> dim;
  bool keep_dim{false};
};

// For Stack Op
struct StackParam : ParamBase {
  std::vector<lite::Tensor*> X;
  lite::Tensor* Out{};

  int axis{0};
};

// For Power Op
struct PowerParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};

  float scale{};
  float shift{};
  float power{};
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
};

// For Scale Op
struct ScaleParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* output{};

  float scale{1.};
  float bias{};
  bool bias_after_scale{true};
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({x}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

// For Softmax op
struct SoftmaxParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* output{};
  int axis{-1};
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({x}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
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
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({x}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

// For Concat op
struct ConcatParam : ParamBase {
  std::vector<lite::Tensor*> x{};
  lite::Tensor* output{};
  int axis{0};
  lite::Tensor* axis_tensor{};
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      std::vector<const Tensor*> vec;
      for (auto in : x) {
        vec.push_back(in);
      }
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>(vec));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

/// ----------------------- activation operators ----------------------
struct ActivationParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite_api::ActivationType active_type;
  bool has_active{false};
  float Leaky_relu_alpha{0};   // leaky_relu param
  float Relu_clipped_coef{6};  // relu_clipped param
  std::string Prelu_mode{
      "channel"};  // prelu param, can be "all", "channel" or "element"
  lite::Tensor* Prelu_alpha{};  // prelu param
  float Swish_beta;             // swish param
  // hard_sigmoid param
  float hard_sigmoid_slope{0.2f};
  float hard_sigmoid_offset{0.5f};
  // hard_swish param
  float hard_swish_threshold{6.0};
  float hard_swish_scale{6.0};
  float hard_swish_offset{3.0};
};

struct ActivationGradParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Out{};
  // for backward
  lite::Tensor* X_grad{};
  const lite::Tensor* Out_grad{};
};

// For Convolution op
struct ConvParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* filter{};
  lite::Tensor* bias{nullptr};
  lite::Tensor* residualData{nullptr};
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
  // support var_length or not
  bool var_length{false};
  // only used in conv_transpose.
  std::vector<int> output_size;
  // for int8
  WITH_INT8_CONFIG

  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({x}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
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
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({x}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({y}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

// For Pooling op
struct PoolParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* output{};
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
  std::string data_format{"AnyLayout"};
  // for int8
  WITH_INT8_CONFIG
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({x}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

// For Dropout op
struct DropoutParam : ParamBase {
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
struct SplitParam : ParamBase {
  lite::Tensor* x{};
  std::vector<lite::Tensor*> output{};
  lite::Tensor* axis_tensor;
  std::vector<lite::Tensor*> sections_tensor_list{};

  int axis{-1};
  int num{0};
  std::vector<int> sections;
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({x}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

// For Transpose op
struct TransposeParam : ParamBase {
  const lite::Tensor* x{};
  lite::Tensor* output{};
  lite::Tensor* xshape{};

  std::vector<int> axis;
  bool use_mkldnn{false};
  std::string data_format{"AnyLayout"};
  ///////////////////////////////////////////////////////////////////////////////////
  //  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({x}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

/// ----------------------- element wise operators ----------------------
struct ElementwiseParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  int axis{-1};  // for broadcasting.
  // for int8
  WITH_INT8_CONFIG
  float x_input_scale{1.0};
  float y_input_scale{1.0};
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({X, Y}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({Out}));
    }
    return output_tensor_ptrs_cache_.get();
  }
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

/// ----------------------- fill_constant operators ----------------------
struct FillConstantParam : ParamBase {
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
  std::vector<int64_t> shape{};
  lite::Tensor* shape_tensor{nullptr};
  std::vector<lite::Tensor*> shape_tensor_list{};

  float value{0.0f};
  // useless for x86, keep it for compatibility
  bool force_cpu{false};
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
  std::vector<int64_t> shape{};
  float min{-1.0f};
  float max{1.0f};
  int seed{0};
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
  lite::Tensor* Out{};
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
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  std::vector<int> offsets;
  std::vector<int> shape;
};

///----------------------- argmax operators ----------------------
struct ArgmaxParam : ParamBase {
  lite::Tensor* X{};
  lite::Tensor* Out{};
  int Axis{0};
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
};

/// ----------------------- priorbox operators ----------------------
struct PriorBoxParam : ParamBase {
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
  ///////////////////////////////////////////////////////////////////////////////////
  //  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({X}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({Out}));
    }
    return output_tensor_ptrs_cache_.get();
  }
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
  lite::Tensor* Out{};
};

struct WhileParam : ParamBase {
  Scope* scope{};
  Tensor* cond{};
  cpp::BlockDesc* sub_block{};
  std::vector<Tensor*> x{};
  std::vector<Tensor*> outs{};
};

struct TopkParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* Indices{};
  int K{1};
};

struct IncrementParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  float step{1};
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
  std::string pool_type{"AVERAGE"};
#ifdef LITE_WITH_X86
  float pad_value{0.0};
  lite::Tensor* MaxIndex{};
#endif
};

struct SequenceConvParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Filter{};
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

struct SequenceUnpadParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Length{};
  lite::Tensor* Out{};
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

struct ReduceMaxParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  std::vector<int> dim{};
  bool keep_dim{false};
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
  lite::Tensor* x{};
  lite::Tensor* output{};
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
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  std::vector<int> axes{};
  std::vector<int> starts{};
  std::vector<int> ends{};
  std::vector<int> decrease_axis{};
  std::vector<int> infer_flags{};
  std::vector<lite::Tensor*> StartsTensorList{};
  std::vector<lite::Tensor*> EndsTensorList{};
  lite::Tensor* StartsTensor{nullptr};
  lite::Tensor* EndsTensor{nullptr};
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({X}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({Out}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct AffineChannelParam : ParamBase {
  const lite::Tensor* X{};  // X is 4D tensor
  const lite::Tensor* Scale{};
  const lite::Tensor* Bias{};
  std::string data_layout{"NCHW"};  // optional string from: NHWC, NCHW.
  lite::Tensor* Out{};
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
};
/// ----------------------- squeeze operators ----------------------
struct SqueezeParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* XShape{};
  std::vector<int> axes{};
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({X}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({Out}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct UnsqueezeParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  lite::Tensor* XShape{};
  std::vector<int> axes{};
  const lite::Tensor* axes_tensor{};
  std::vector<const lite::Tensor*> axes_tensor_vct{};
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({X}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({Out}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

/// ----------------------- expand operators ----------------------
struct ExpandParam : ParamBase {
  const lite::Tensor* X{};
  lite::Tensor* Out{};
  std::vector<int> expand_times{};
};

/// ----------------------- matmul operators ----------------------
struct MatMulParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Y{};
  lite::Tensor* Out{};
  bool transpose_X{false};
  bool transpose_Y{false};
  float alpha{1.0f};
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>({X, Y}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(new std::vector<lite::Tensor*>({Out}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct GatherParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* Index{};
  lite::Tensor* Out{};
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
  lite::Tensor* X{};
  lite::Tensor* ROIs{};
  lite::Tensor* Out{};
  float spatial_scale{1.0};
  int pooled_height{1};
  int pooled_width{1};
  int sampling_ratio{-1};
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

/// --------------- search_fc operators ------------------
struct SearchFcParam : ParamBase {
  const lite::Tensor* X{};
  const lite::Tensor* W{};
  const lite::Tensor* b{};
  lite::Tensor* Out{};
  int out_size{};
};
/// --------------------- match_matrix_tensor operators --------------------
struct MatchMatrixTensorParam : ParamBase {
  const lite::Tensor* x{};
  const lite::Tensor* y{};
  const lite::Tensor* w{};
  lite::Tensor* out{};
  lite::Tensor* tmp{};

  int dim_t;
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
  std::vector<lite::Tensor*> x{};
  std::vector<lite::Tensor*> outs{};
  cpp::BlockDesc* sub_block{};
  Scope* scope{};
  bool is_scalar_condition{};
};

struct CollectFpnProposalsParam : ParamBase {
  std::vector<lite::Tensor*> multi_level_rois{};
  std::vector<lite::Tensor*> multi_level_scores{};
  lite::Tensor* fpn_rois{};
  int post_nms_topN{};
};

struct DistributeFpnProposalsParam : ParamBase {
  const lite::Tensor* fpn_rois{};
  std::vector<lite::Tensor*> multi_fpn_rois{};
  lite::Tensor* restore_index{};
  int min_level{};
  int max_level{};
  int refer_level{};
  int refer_scale{};
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
};
/// --------------------- grid sampler operators --------------------
struct GridSamplerParam : ParamBase {
  lite::Tensor* x{};
  lite::Tensor* out{};
  lite::Tensor* grid{};
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
  std::string gate_activation;
  std::string cell_activation;
  std::string candidate_activation;
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

struct XPUMultiEncoderParam : ParamBase {
  lite::Tensor* input{};
  std::vector<lite::Tensor*> fc_weight;
  std::vector<lite::Tensor*> fc_bias;
  std::vector<lite::Tensor*> ln_scale;
  std::vector<lite::Tensor*> ln_bias;
  lite::Tensor* fc_weight_max{};
  lite::Tensor* mask{};
  lite::Tensor* output{};

  int n_layers{};
  int head_num{};
  int size_per_head{};
  std::string act_type{};
};

struct XPUEmbeddingWithEltwiseAddParam : ParamBase {
  std::vector<lite::Tensor*> Ids;
  std::vector<lite::Tensor*> Tables;
  lite::Tensor* Out{};
  int64_t padding_idx{-1};
};

struct XPUFcParam : ParamBase {
  lite::Tensor* input{nullptr};
  lite::Tensor* w{nullptr};
  lite::Tensor* bias{nullptr};
  lite::Tensor* output{nullptr};

  int in_num_col_dims{1};
  lite::DDim in_mat_dims;
  float w_max{0.0f};
  bool transpose_w{true};
  std::string activation_type{""};
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
