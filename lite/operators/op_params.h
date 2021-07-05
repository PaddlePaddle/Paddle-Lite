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
#include "lite/model_parser/base/apis.h"
#include "lite/model_parser/cpp_desc.h"
#include "lite/utils/all.h"
/*
 * This file contains all the argument parameter data structure for operators.
 */

namespace paddle {
namespace lite_metal {
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
  std::vector<lite_metal::Tensor>* feed_list{};
  lite_metal::Tensor* out{};
  int col;
};

struct FetchParam : ParamBase {
  const lite_metal::Tensor* input{};
  std::vector<lite_metal::Tensor>* fetch_list{};
  int col;
};

// Helper op for lite framework
struct IoCopyParam : ParamBase {
  const lite_metal::Tensor* x{nullptr};
  const std::vector<lite_metal::Tensor>* x_array{nullptr};
  lite_metal::Tensor* y{nullptr};
  std::vector<lite_metal::Tensor>* y_array{nullptr};
  int process_type{0};
};

struct LayoutParam : ParamBase {
  const lite_metal::Tensor* x{};
  lite_metal::Tensor* y{};
  int process_type{0};
};

struct CalibParam : ParamBase {
  const lite_metal::Tensor* input{};
  lite_metal::Tensor* output{};
  float scale;
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct SubgraphParam : ParamBase {
  std::vector<std::string> input_names{};
  std::vector<std::string> output_names{};
  std::vector<std::string> input_data_names{};
  std::vector<std::string> output_data_names{};
  int block_idx{-1};
  std::shared_ptr<const cpp::ProgramDesc> program_desc{nullptr};
  Scope* exec_scope{nullptr};
};

/// -------------------------- NN operators ------------------------------------

struct FcParam : ParamBase {
  lite_metal::Tensor* input{nullptr};
  lite_metal::Tensor* w{nullptr};
  lite_metal::Tensor* bias{nullptr};
  lite_metal::Tensor* Prelu_alpha{nullptr};
  lite_metal::Tensor* output{nullptr};
  lite_metal::DDim in_mat_dims;
  // original dims of input weight
  lite_metal::DDim w_dims;
  int in_num_col_dims{1};
  std::string activation_type{""};
  bool padding_weights{false};
  std::string Prelu_mode{
      "channel"};  // prelu param, can be "all", "channel" or "element"
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct SearchSeqFcParam : ParamBase {
  lite_metal::Tensor* x{nullptr};
  lite_metal::Tensor* w{nullptr};
  lite_metal::Tensor* b{nullptr};
  lite_metal::Tensor* out{nullptr};
  int out_size;
};

// For Interpolate Op
struct InterpolateParam : ParamBase {
  lite_metal::Tensor* X{};
  lite_metal::Tensor* OutSize{};
  lite_metal::Tensor* Out{};
  std::vector<const lite_metal::Tensor*> SizeTensor;
  lite_metal::Tensor* Scale{};

  float scale{0.f};
  std::vector<float> scale_v{};
  int out_h{-1};
  int out_w{-1};
  bool align_corners{true};
  int align_mode{1};
  std::string interp_method{"Nearest"};
  DataLayoutType data_layout{DATALAYOUT(kNCHW)};
};

// For Mul Op
struct MulParam : ParamBase {
  const lite_metal::Tensor* x{};
  const lite_metal::Tensor* y{};
  lite_metal::Tensor* output{};

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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct MulGradParam : ParamBase {
  const lite_metal::Tensor* x{};
  const lite_metal::Tensor* y{};
  const lite_metal::Tensor* output_grad{};
  lite_metal::Tensor* x_grad{};
  lite_metal::Tensor* y_grad{};

  int x_num_col_dims{1};
  int y_num_col_dims{1};
};

// For Stack Op
struct StackParam : ParamBase {
  std::vector<lite_metal::Tensor*> X;
  lite_metal::Tensor* Out{};

  int axis{0};
};

// For Unstack Op
struct UnstackParam : ParamBase {
  const lite_metal::Tensor* X{nullptr};
  std::vector<lite_metal::Tensor*> Out{};

  int axis{0};
  int num{1};
};

// For Power Op
struct PowerParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};

  float scale{};
  float shift{};
  float power{};
};

// For Pow Op
struct PowParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};

  float factor{1.f};
};

// For Sign Op
struct SignParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
};

struct ShuffleChannelParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};

  int group;
};

// For Yolobox
struct YoloBoxParam : ParamBase {
  lite_metal::Tensor* X{};
  lite_metal::Tensor* ImgSize{};
  lite_metal::Tensor* Boxes{};
  lite_metal::Tensor* Scores{};

  std::vector<int> anchors{};
  int class_num{0};
  float conf_thresh{0.f};
  int downsample_ratio{0};
  bool clip_bbox{true};
  float scale_x_y{1.0f};
};

// For Scale Op
struct ScaleParam : ParamBase {
  lite_metal::Tensor* x{};
  lite_metal::Tensor* output{};

  float scale{1.f};
  float bias{0.f};
  bool bias_after_scale{true};
  std::string activation_type{""};
  bool fuse_relu{false};
  float alpha{6.f};

  bool fuse_scaleact{false};
  float scale1{1.f};
  float bias1{0.f};
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

// For Scatter OP
struct ScatterParam : ParamBase {
  lite_metal::Tensor* x{};
  lite_metal::Tensor* indexs{};
  lite_metal::Tensor* updates{};
  lite_metal::Tensor* output{};

  bool overwrite{true};
};

// For Softmax op
struct SoftmaxParam : ParamBase {
  lite_metal::Tensor* x{};
  lite_metal::Tensor* output{};
  int axis{-1};
  bool use_cudnn{true};
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

// For Reshape and Reshape2 Op
struct ReshapeParam : ParamBase {
  const lite_metal::Tensor* x{};
  std::vector<const lite_metal::Tensor*> shape_tensor_vct{};
  const lite_metal::Tensor* shape_tensor{};
  std::vector<int> shape_vct{};
  lite_metal::Tensor* output{};

  lite_metal::Tensor* xshape{};
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }

#ifdef LITE_WITH_METAL
  std::vector<int> excepted_transpose_;
#endif
};

// For Concat op
struct ConcatParam : ParamBase {
  std::vector<lite_metal::Tensor*> x{};
  lite_metal::Tensor* output{};
  int axis{0};
  lite_metal::Tensor* axis_tensor{};
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

/// ----------------------- activation operators ----------------------
struct ActivationParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
  lite_metal_api::ActivationType active_type{lite_metal_api::ActivationType::kIndentity};
  bool has_active{false};
  float Leaky_relu_alpha{0.f};   // leaky_relu param
  float Relu_clipped_coef{6.f};  // relu_clipped param
  std::string Prelu_mode{
      "channel"};  // prelu param, can be "all", "channel" or "element"
  lite_metal::Tensor* Prelu_alpha{};  // prelu param
  float Swish_beta;             // swish param
  // hard_sigmoid param
  float hard_sigmoid_slope{0.2f};
  float hard_sigmoid_offset{0.5f};
  // hard_swish param
  float hard_swish_threshold{6.0f};
  float hard_swish_scale{6.0f};
  float hard_swish_offset{3.0f};
  // thresholded_relu
  float relu_threshold{1.0f};
  // elu
  float Elu_alpha{1.0f};
  // relu6
  float threshold{6.0f};
  // gelu
  bool gelu_approximate{false};

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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({Out}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct ActivationGradParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Out{};
  // for backward
  lite_metal::Tensor* X_grad{};
  const lite_metal::Tensor* Out_grad{};
};

// For Convolution op
struct ConvParam : ParamBase {
  lite_metal::Tensor* x{};
  lite_metal::Tensor* filter{};
  lite_metal::Tensor* bias{nullptr};
  lite_metal::Tensor* residualData{nullptr};
  lite_metal::Tensor* second_x{nullptr};
  lite_metal::Tensor* output{};
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
  // for elementwise tree fuse
  std::string fuse_elementwise_op_type{""};
  // support var_length or not
  bool var_length{false};
  // only used in conv_transpose.
  std::vector<int> output_size;
  std::vector<int> output_padding;

#ifdef LITE_WITH_FPGA
  lite_metal::Tensor* scale{nullptr};
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

// For BatchNorm op
struct BatchNormParam : ParamBase {
  lite_metal::Tensor* x{};
  lite_metal::Tensor* bias{};
  lite_metal::Tensor* scale{};
  lite_metal::Tensor* mean{};
  lite_metal::Tensor* variance{};
  lite_metal::Tensor* y{};
  lite_metal::Tensor* mean_out{};
  lite_metal::Tensor* variance_out{};
  lite_metal::Tensor* saved_mean{};
  lite_metal::Tensor* saved_variance{};
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({y}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

// For Pooling op
struct PoolParam : ParamBase {
  lite_metal::Tensor* x{};
  lite_metal::Tensor* output{};
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

// For Dropout op
struct DropoutParam : ParamBase {
  const lite_metal::Tensor* x{};
  lite_metal::Tensor* output{};
  lite_metal::Tensor* mask{};
  float dropout_prob{.5f};
  bool is_test{false};
  bool fix_seed{false};
  int seed{0};
  std::string dropout_implementation{"downgrade_in_infer"};
};

// For PadConstantLike op
struct PadConstantLikeParam : ParamBase {
  const lite_metal::Tensor* x{};
  const lite_metal::Tensor* y{};
  lite_metal::Tensor* output{};
  float pad_value{0.0f};
};

// For Split op
struct SplitParam : ParamBase {
  const lite_metal::Tensor* x{nullptr};
  std::vector<lite_metal::Tensor*> output{};
  const lite_metal::Tensor* axis_tensor{nullptr};
  std::vector<lite_metal::Tensor*> sections_tensor_list{};

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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct UnbindParam : ParamBase {
  lite_metal::Tensor* x{};
  std::vector<lite_metal::Tensor*> output{};

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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

// For Transpose op
struct TransposeParam : ParamBase {
  const lite_metal::Tensor* x{};
  lite_metal::Tensor* output{};
  lite_metal::Tensor* xshape{};

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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct TrilTriuParam : ParamBase {
  const lite_metal::Tensor* x{nullptr};
  lite_metal::Tensor* out{nullptr};

  int diagonal{0};
  bool lower{true};
};

/// ----------------------- element wise operators ----------------------
struct ElementwiseParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Y{};
  lite_metal::Tensor* Out{};
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({Out}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct ElementwiseGradParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Y{};
  const lite_metal::Tensor* OutGrad{};
  lite_metal::Tensor* XGrad{};
  lite_metal::Tensor* YGrad{};
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
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
};

struct MeanGradParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Out_grad{};
  // for backward
  lite_metal::Tensor* X_grad{};
};

struct FillAnyLikeParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
  float value{0.0f};
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
};

/// ----------------------- fill_constant operators ----------------------
struct FillConstantParam : ParamBase {
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
  std::vector<int64_t> shape{};
  lite_metal::Tensor* shape_tensor{nullptr};
  lite_metal::Tensor* value_tensor{nullptr};
  std::vector<lite_metal::Tensor*> shape_tensor_list{};

  float value{0.0f};
  // useless for x86, keep it for compatibility
  bool force_cpu{false};
  lite_metal::Tensor* in{};
  lite_metal::Tensor* out{};
};

struct FillConstantBatchSizeLikeParam : ParamBase {
  const lite_metal::Tensor* input{nullptr};
  lite_metal::Tensor* out{nullptr};

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
  const lite_metal::Tensor* x{};
  const lite_metal::Tensor* in_scale{};
  const lite_metal::Tensor* in_accum{};
  const lite_metal::Tensor* in_state{};
  lite_metal::Tensor* out{};
  lite_metal::Tensor* out_scale{};
  lite_metal::Tensor* out_state{};
  lite_metal::Tensor* out_accum{};
  int bit_length;
  bool is_test{true};
  float moving_rate{0.9f};
};

struct FakeDequantizeMaxAbsParam : ParamBase {
  const lite_metal::Tensor* x{};
  const lite_metal::Tensor* in_scale{};
  lite_metal::Tensor* out{};
  float max_range;
};

struct FakeChannelWiseDequantizeMaxAbsParam : ParamBase {
  const lite_metal::Tensor* x{};
  std::vector<const lite_metal::Tensor*> scale_tensors{};
  lite_metal::Tensor* out{};
  std::vector<int> quant_bits;
};

struct FakeQuantDequantAbsMaxParam : ParamBase {
  const lite_metal::Tensor* x{};
  lite_metal::Tensor* out{};
  lite_metal::Tensor* out_scale{};
  int bit_length;
};

struct FakeChannelWiseQuantDequantAbsMaxParam : ParamBase {
  const lite_metal::Tensor* x{};
  lite_metal::Tensor* out{};
  lite_metal::Tensor* out_scale{};
  int quant_axis;
  int bit_length;
};

/// ----------------------- sgd operators ----------------------
struct SGDParam : ParamBase {
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};

  const lite_metal::Tensor* Param{};
  const lite_metal::Tensor* LearningRate{};
  const lite_metal::Tensor* Grad{};
  lite_metal::Tensor* ParamOut{};
};

/// ----------------------- uniform_random operators ----------------------
struct UniformRandomParam : ParamBase {
  const lite_metal::Tensor* shape_tensor{nullptr};
  std::vector<lite_metal::Tensor*> shape_tensor_list{};
  std::vector<int64_t> shape{};
  float min{-1.0f};
  float max{1.0f};
  int seed{0};
  int dtype{static_cast<int>(VarDescAPI::VarDataType::FP32)};
  lite_metal::Tensor* Out{};
};
/// ----------------------- negative operators --------------
struct NegativeParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
};
/// ----------------------- pad2d operators ----------------------
struct Pad2dParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
  std::vector<int> paddings{0, 0, 0, 0};
  std::string mode{"constant"};
  float pad_value = 0.f;
  std::string data_format{"NCHW"};
};

/// ----------------------- Crop operators ----------------------
struct CropParam : ParamBase {
  const lite_metal::Tensor* X{nullptr};
  const lite_metal::Tensor* Y{nullptr};
  const lite_metal::Tensor* Offsets{nullptr};
  lite_metal::Tensor* Out{nullptr};
  std::vector<int> offsets;
  std::vector<int> shape;
};

/// ----------------------- CropTensor operators ----------------------
struct CropTensorParam : ParamBase {
  const lite_metal::Tensor* X{nullptr};
  const lite_metal::Tensor* Shape{nullptr};
  const lite_metal::Tensor* Offsets{nullptr};
  const std::vector<lite_metal::Tensor>* ShapeTensor{nullptr};
  const std::vector<lite_metal::Tensor>* OffsetsTensor{nullptr};
  lite_metal::Tensor* Out{nullptr};
  std::vector<int> offsets;
  std::vector<int> shape;
};

///----------------------- argmax operators ----------------------
struct ArgmaxParam : ParamBase {
  lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
  int Axis{0};
  int dtype{-1};
  bool keepdims{false};
};

///----------------------- axpy operators ----------------------
struct AxpyParam : ParamBase {
  lite_metal::Tensor* Scale{};
  lite_metal::Tensor* X{};
  lite_metal::Tensor* Bias{};
  lite_metal::Tensor* Out{};
};
/// ----------------------- GRU unit operators ----------------------f
struct GRUUnitParam : ParamBase {
  enum ActType { identity, sigmoid, tanh, relu };
  const lite_metal::Tensor* input{nullptr};
  const lite_metal::Tensor* hidden_prev{nullptr};
  const lite_metal::Tensor* weight{nullptr};
  const lite_metal::Tensor* bias{nullptr};
  lite_metal::Tensor* gate{nullptr};
  lite_metal::Tensor* reset_hidden_prev{nullptr};
  lite_metal::Tensor* hidden{nullptr};

  int gate_activation{ActType::sigmoid};
  int activation{ActType::tanh};
  bool origin_mode{false};
};

/// ------------------------------ lrn operators ------------------------------
struct LrnParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
  int n{5};
  float alpha{1e-4f};
  float beta{0.75f};
  float k{1.f};
  std::string norm_region{"AcrossChannels"};
};

/// ----------------------- decode_bboxes operators ----------------------
struct DecodeBboxesParam : ParamBase {
  const lite_metal::Tensor* loc_data{};
  const lite_metal::Tensor* prior_data{};
  lite_metal::Tensor* bbox_data{};

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
  const lite_metal::Tensor* prior_box{};
  const lite_metal::Tensor* prior_box_var{};
  const lite_metal::Tensor* target_box{};
  lite_metal::Tensor* proposals{};
  // code_type: encode_center_size and decode_center_size
  std::string code_type{"encode_center_size"};
  bool box_normalized{true};
  int axis{0};
  std::vector<float> variance{};
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(new std::vector<const Tensor*>(
          {prior_box, prior_box_var, target_box}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(
          new std::vector<lite_metal::Tensor*>({proposals}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

/// ----------------------- multiclass_nms operators ----------------------
struct MulticlassNmsParam : ParamBase {
  const lite_metal::Tensor* bboxes{};
  const lite_metal::Tensor* scores{};
  lite_metal::Tensor* out{};
  lite_metal::Tensor* index{};
  int background_label{0};
  float score_threshold{};
  int nms_top_k{};
  float nms_threshold{0.3f};
  float nms_eta{1.0f};
  int keep_top_k;
  bool normalized{true};
  const lite_metal::Tensor* rois_num{};
  lite_metal::Tensor* nms_rois_num{};
};

/// ----------------------- matrix_nms operators ----------------------
struct MatrixNmsParam : ParamBase {
  const lite_metal::Tensor* bboxes{};
  const lite_metal::Tensor* scores{};
  lite_metal::Tensor* out{};
  lite_metal::Tensor* index{};
  lite_metal::Tensor* rois_num{};
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
  lite_metal::Tensor* input{};
  lite_metal::Tensor* image{};
  lite_metal::Tensor* boxes{};
  lite_metal::Tensor* variances{};

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
  // priortype: prior_min, prior_max, prior_com
  std::vector<std::string> order;
  bool min_max_aspect_ratios_order{false};
  ///////////////////////////////////////////////////////////////////////////////////
  // get a vector of input tensors
  const std::vector<const Tensor*>* input_tensor_ptrs() override {
    if (!input_tensor_ptrs_cache_) {
      input_tensor_ptrs_cache_.reset(
          new std::vector<const Tensor*>({input, image}));
    }
    return input_tensor_ptrs_cache_.get();
  }
  // get a vector of output tensors
  std::vector<Tensor*>* output_tensor_ptrs() override {
    if (!output_tensor_ptrs_cache_) {
      output_tensor_ptrs_cache_.reset(
          new std::vector<lite_metal::Tensor*>({boxes, variances}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct DensityPriorBoxParam : public PriorBoxParam {
  std::vector<float> fixed_sizes;
  std::vector<float> fixed_ratios;
  std::vector<int> density_sizes;
};
/// ----------------------- GRU operators ----------------------f
struct GRUParam : ParamBase {
  const lite_metal::Tensor* input{nullptr};
  const lite_metal::Tensor* h0{nullptr};
  const lite_metal::Tensor* weight{nullptr};
  const lite_metal::Tensor* bias{nullptr};
  lite_metal::Tensor* batch_gate{nullptr};
  lite_metal::Tensor* batch_reset_hidden_prev{nullptr};
  lite_metal::Tensor* batch_hidden{nullptr};
  lite_metal::Tensor* hidden{nullptr};

  std::string gate_activation{"sigmoid"};
  std::string activation{"tanh"};
  bool is_reverse{false};
  bool origin_mode{false};

  // for int8
  WITH_INT8_CONFIG
};

struct BiGRUParam : ParamBase {
  const lite_metal::Tensor* input{nullptr};
  const lite_metal::Tensor* fw_mul_w{nullptr};
  const lite_metal::Tensor* fw_mul_b{nullptr};
  const lite_metal::Tensor* fw_gru_w{nullptr};
  const lite_metal::Tensor* fw_gru_b{nullptr};
  const lite_metal::Tensor* bw_mul_w{nullptr};
  const lite_metal::Tensor* bw_mul_b{nullptr};
  const lite_metal::Tensor* bw_gru_w{nullptr};
  const lite_metal::Tensor* bw_gru_b{nullptr};
  lite_metal::Tensor* fw_output{nullptr};
  lite_metal::Tensor* bw_output{nullptr};

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
  std::vector<lite_metal::Tensor>* ids{nullptr};
  std::vector<lite_metal::Tensor>* scores{nullptr};
  lite_metal::Tensor* sentence_ids{nullptr};
  lite_metal::Tensor* sentence_scores{nullptr};
  int beam_size;
  int end_id;
};

/// ----------------------- LookupTable operators ----------------------f
struct LookupTableParam : ParamBase {
  const lite_metal::Tensor* W{nullptr};
  const lite_metal::Tensor* Ids{nullptr};
  lite_metal::Tensor* Out{nullptr};
  int64_t padding_idx{-1};
  bool is_test{true};
  std::string entry_config{""};  // used in distributed training
  std::string entry{"none"};
};

struct LookupTableDequantParam : ParamBase {
  lite_metal::Tensor* W{nullptr};
  lite_metal::Tensor* Ids{nullptr};
  lite_metal::Tensor* Out{nullptr};
  int64_t padding_idx{-1};
};

struct Im2SequenceParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Y{};
  lite_metal::Tensor* Out{};
  std::vector<int> kernels{3, 3};
  std::vector<int> strides{1, 1};
  std::vector<int> paddings{0, 0, 0, 0};
  std::vector<int> out_strides{1, 1};
};

struct SequenceSoftmaxParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({Out}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct NormParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
  lite_metal::Tensor* Norm{};
  int axis{1};
  float epsilon{1e-10f};
};
struct LayerNormParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Scale{};
  const lite_metal::Tensor* Bias{};
  lite_metal::Tensor* Y{};
  lite_metal::Tensor* Mean{};
  lite_metal::Tensor* Variance{};
  int begin_norm_axis{1};
  float epsilon{1e-5f};
};

struct LogicalParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Y{};
  lite_metal::Tensor* Out{};
};

struct CompareParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Y{};
  bool force_cpu{0};
  int axis{-1};
  lite_metal::Tensor* Out{};
};

struct WhileParam : ParamBase {
  Tensor* cond{};
  int block_idx{-1};
  std::shared_ptr<const cpp::ProgramDesc> program_desc{nullptr};
  Scope* exec_scope{nullptr};
};

struct TopkParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* KTensor{};
  lite_metal::Tensor* Out{};
  lite_metal::Tensor* Indices{};
  bool k_is_tensor{false};
  int K{1};
  int axis{-1};
};

struct IncrementParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
  float step{1.f};
};

struct WriteToArrayParam : ParamBase {
  const lite_metal::Tensor* X{nullptr};
  const lite_metal::Tensor* I{nullptr};
  std::vector<lite_metal::Tensor>* Out{nullptr};
};

struct ReadFromArrayParam : ParamBase {
  const std::vector<lite_metal::Tensor>* X{nullptr};
  const lite_metal::Tensor* I{nullptr};
  lite_metal::Tensor* Out{nullptr};
};

struct BeamSearchParam : ParamBase {
  const lite_metal::Tensor* pre_ids{};
  const lite_metal::Tensor* pre_scores{};
  const lite_metal::Tensor* ids{};
  const lite_metal::Tensor* scores{};
  lite_metal::Tensor* selected_ids{};
  lite_metal::Tensor* selected_scores{};
  lite_metal::Tensor* parent_idx{};
  int level;
  int beam_size;
  int end_id;
  bool is_accumulated;
};

struct SequencePoolParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
  lite_metal::Tensor* MaxIndex{};
  std::string pool_type{"AVERAGE"};
  float pad_value{0.0f};
};

struct SequenceConvParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Filter{};
  lite_metal::Tensor* Out{};
  int contextStart{0};
  int contextStride{1};
  int contextLength;
};

struct SequencePoolConcatParam : ParamBase {
  std::vector<lite_metal::Tensor*> X{};
  lite_metal::Tensor* Out{};
  std::vector<std::string> pool_type{};
};

struct SequencePoolGradParam : ParamBase {
  const lite_metal::Tensor* X{};
  std::string pool_type{"AVERAGE"};
#ifdef LITE_WITH_X86
  float pad_value{0.0f};
#endif
  // for backward
  const lite_metal::Tensor* Out_Grad{};
  const lite_metal::Tensor* MaxIndex_Grad{};
  lite_metal::Tensor* X_Grad{};
};

struct SearchGroupPaddingParam : ParamBase {
  lite_metal::Tensor* x{};
  lite_metal::Tensor* out_emb_padding{};
  lite_metal::Tensor* out_new{};
  lite_metal::Tensor* out_padding{};
  int pad_id;
};

struct SequenceReshapeParam : ParamBase {
  lite_metal::Tensor* x{};
  lite_metal::Tensor* output{};
  int new_dim;
};

struct SequenceExpandParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Y{};
  lite_metal::Tensor* Out{};
  int ref_level{-1};
};

struct SequencePadParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* PadValue{};
  lite_metal::Tensor* Out{};
  lite_metal::Tensor* Length{};
  int padded_length{-1};
};

struct SequenceUnpadParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Length{};
  lite_metal::Tensor* Out{};
};

struct SequenceMaskParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* MaxLenTensor{nullptr};
  lite_metal::Tensor* Y{};
  int maxlen{-1};
  int out_dtype;
};

struct SequenceExpandAsParam : ParamBase {
  const lite_metal::Tensor* x{nullptr};
  const lite_metal::Tensor* y{nullptr};
  lite_metal::Tensor* out{nullptr};
};

struct SequenceReverseParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
};

struct SequenceConcatParam : ParamBase {
  std::vector<lite_metal::Tensor*> X{};
  lite_metal::Tensor* Out{};
};

struct MeshgridParam : ParamBase {
  std::vector<lite_metal::Tensor*> X{};
  std::vector<lite_metal::Tensor*> Out{};
};

struct AttentionPaddingMaskParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Y{};
  int pad_id;
  float mask;
  lite_metal::Tensor* Out{};
  lite_metal::Tensor* pad_begin{};
};

struct SequenceArithmeticParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Y{};
  int op_type{1};
  lite_metal::Tensor* Out{};
};

struct LodResetParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Y{};
  lite_metal::Tensor* Out{};
  std::vector<int> target_lod;
  bool append;
};

struct IsEmptyParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
};

struct ReduceParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
  std::vector<int> dim{0};
  bool keep_dim{false};
  bool reduce_all{false};
};

struct VarConv2DParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* ROW{};
  const lite_metal::Tensor* COLUMN{};
  const lite_metal::Tensor* W{};
  lite_metal::Tensor* Out{};
  lite_metal::Tensor* Col{};

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
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
};

struct CastParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
  int out_dtype{2};
  int in_dtype{2};
};

struct SliceParam : ParamBase {
  const lite_metal::Tensor* X{nullptr};
  lite_metal::Tensor* Out{nullptr};
  std::vector<int> axes{};
  std::vector<int> starts{};
  std::vector<int> ends{};
  std::vector<int> decrease_axis{};
  std::vector<int> infer_flags{};
  std::vector<lite_metal::Tensor*> StartsTensorList{};
  std::vector<lite_metal::Tensor*> EndsTensorList{};
  const lite_metal::Tensor* StartsTensor{nullptr};
  const lite_metal::Tensor* EndsTensor{nullptr};
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({Out}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct AffineChannelParam : ParamBase {
  const lite_metal::Tensor* X{};  // X is 4D tensor
  const lite_metal::Tensor* Scale{};
  const lite_metal::Tensor* Bias{};
  std::string data_layout{"NCHW"};  // optional string from: NHWC, NCHW.
  lite_metal::Tensor* Out{};
};

struct AffineGridParam : ParamBase {
  const lite_metal::Tensor* X{};  // Theta:shape {?, 2, 3}
  std::vector<int> output_shape;
  const lite_metal::Tensor* OutputShape;
  lite_metal::Tensor* Out{};
  bool align_corners{true};
};

struct AnchorGeneratorParam : ParamBase {
  const lite_metal::Tensor* Input{};
  std::vector<float> anchor_sizes{};
  std::vector<float> aspect_ratios{};
  std::vector<float> stride{};
  std::vector<float> variances{{0.1f, 0.1f, 0.2f, 0.2f}};
  float offset{0.5f};

  lite_metal::Tensor* Anchors{};
  lite_metal::Tensor* Variances{};
};

struct GenerateProposalsParam : ParamBase {
  // inputs
  const lite_metal::Tensor* Scores{};
  const lite_metal::Tensor* BboxDeltas{};
  const lite_metal::Tensor* ImInfo{};
  lite_metal::Tensor* Anchors{};
  lite_metal::Tensor* Variances{};

  // attrs
  int pre_nms_topN{6000};
  int post_nms_topN{1000};
  float nms_thresh{0.5f};
  float min_size{0.1f};
  float eta{1.0f};

  // outputs
  lite_metal::Tensor* RpnRois{};
  lite_metal::Tensor* RpnRoiProbs{};
  lite_metal::Tensor* RpnRoisLod{};
  lite_metal::Tensor* RpnRoisNum{};
};

struct GenerateProposalsV2Param : ParamBase {
  // inputs
  const lite_metal::Tensor* Scores{};
  const lite_metal::Tensor* BboxDeltas{};
  const lite_metal::Tensor* ImShape{};
  lite_metal::Tensor* Anchors{};
  lite_metal::Tensor* Variances{};

  // attrs
  int pre_nms_topN{6000};
  int post_nms_topN{1000};
  float nms_thresh{0.5f};
  float min_size{0.1f};
  float eta{1.0f};

  // outputs
  lite_metal::Tensor* RpnRois{};
  lite_metal::Tensor* RpnRoiProbs{};
  lite_metal::Tensor* RpnRoisLod{};
  lite_metal::Tensor* RpnRoisNum{};
};

/// ----------------------- squeeze operators ----------------------
struct SqueezeParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
  lite_metal::Tensor* XShape{};
  std::vector<int> axes{};
  bool inplace{false};
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({Out}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct UnsqueezeParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
  lite_metal::Tensor* XShape{};
  std::vector<int> axes{};
  const lite_metal::Tensor* axes_tensor{};
  std::vector<const lite_metal::Tensor*> axes_tensor_vct{};
  bool inplace{false};
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({Out}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

/// ----------------------- expand operators ----------------------
struct ExpandParam : ParamBase {
  const lite_metal::Tensor* X{nullptr};
  const lite_metal::Tensor* ExpandTimes{nullptr};
  std::vector<lite_metal::Tensor*> expand_times_tensor{};
  lite_metal::Tensor* Out{nullptr};
  std::vector<int> expand_times{};
};

/// ----------------------- expand v2 operators ----------------------
struct ExpandV2Param : ParamBase {
  const lite_metal::Tensor* X{nullptr};
  const lite_metal::Tensor* Shape{nullptr};
  std::vector<lite_metal::Tensor*> expand_shapes_tensor{};
  lite_metal::Tensor* Out{nullptr};
  std::vector<int> shape{};
};

/// ----------------------- expand as operators ----------------------
struct ExpandAsParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Target{};
  lite_metal::Tensor* Out{};
};

/// ----------------------- matmul operators ----------------------
struct MatMulParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Y{};
  lite_metal::Tensor* Out{};
  bool transpose_X{false};
  bool transpose_Y{false};
  float alpha{1.0f};
  WITH_INT8_CONFIG
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({Out}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct GatherNdParam : ParamBase {
  const lite_metal::Tensor* x{nullptr};
  const lite_metal::Tensor* index{nullptr};
  lite_metal::Tensor* out{nullptr};
};

struct GatherParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Index{};
  const lite_metal::Tensor* Axis{nullptr};
  lite_metal::Tensor* Out{};
};

struct GatherTreeParam : ParamBase {
  const lite_metal::Tensor* ids{nullptr};
  const lite_metal::Tensor* parents{nullptr};
  lite_metal::Tensor* out{nullptr};
};

/// ----------------------- assign operators -----------------------
struct AssignParam : ParamBase {
  // for tensor
  const lite_metal::Tensor* X{nullptr};
  lite_metal::Tensor* Out{nullptr};

  // for tensor_array
  const std::vector<lite_metal::Tensor>* X_array{nullptr};
  std::vector<lite_metal::Tensor>* Out_array{nullptr};
};

/// ----------------------- roi_align operators -----------------------
struct RoiAlignParam : ParamBase {
  lite_metal::Tensor* X{};
  lite_metal::Tensor* ROIs{};
  lite_metal::Tensor* RoisLod{};
  lite_metal::Tensor* RoisNum{};
  lite_metal::Tensor* Out{};
  float spatial_scale{1.0f};
  int pooled_height{1};
  int pooled_width{1};
  int sampling_ratio{-1};
};

/// ----------------------- box_clip operators -----------------------
struct BoxClipParam : ParamBase {
  const lite_metal::Tensor* Input{};
  const lite_metal::Tensor* ImInfo{};
  lite_metal::Tensor* Output{};
};

struct RangeParam : ParamBase {
  const lite_metal::Tensor* Start;
  const lite_metal::Tensor* End;
  const lite_metal::Tensor* Step;
  lite_metal::Tensor* Out;
};

/// ----------------------- assign_value operators -----------------------
struct AssignValueParam : ParamBase {
  std::vector<int> shape{};
  int dtype{};
  std::vector<float> fp32_values{};
  std::vector<int> int32_values{};
  std::vector<int64_t> int64_values{};
  std::vector<int> bool_values{};
  lite_metal::Tensor* Out{};
};

/// --------------- sequence_topk_avg_pooling operators ------------------
struct SequenceTopkAvgPoolingParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* ROW{};
  const lite_metal::Tensor* COLUMN{};
  lite_metal::Tensor* Out{};
  lite_metal::Tensor* pos{};
  int channel_num{};
  std::vector<int> topks{};
};

/// --------------- topk_pooling operators ------------------
struct TopkPoolingParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* Y{};
  lite_metal::Tensor* Out{};
  int top_k{1};
  int feat_map_num{1};
};

/// --------------- search_fc operators ------------------
struct SearchFcParam : ParamBase {
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* W{};
  const lite_metal::Tensor* b{};
  lite_metal::Tensor* Out{};
  int out_size{};

  bool fuse_relu{false};

#ifdef LITE_WITH_XPU
  bool __xpu__float_to_fix{false};  // Is W already converted to int16/int8
  float __xpu__w_max{0.0f};         // Abs max in W
#endif
};
/// --------------------- match_matrix_tensor operators --------------------
struct MatchMatrixTensorParam : ParamBase {
  const lite_metal::Tensor* x{};
  const lite_metal::Tensor* y{};
  const lite_metal::Tensor* w{};
  lite_metal::Tensor* out{};
  lite_metal::Tensor* tmp{};

  int dim_t;
  bool fuse_relu{false};

#ifdef LITE_WITH_XPU
  bool __xpu__float_to_fix{false};  // Is w already converted to int16/int8
  float __xpu__w_max{0.0f};         // Abs max in w
#endif
};

/// --------------------- search_seq_depadding operators --------------------
struct SearchSeqDepaddingParam : ParamBase {
  const lite_metal::Tensor* pad{};
  const lite_metal::Tensor* src{};
  lite_metal::Tensor* out{};
};

/// --------------------- search_grnn operators --------------------
struct SearchGrnnParam : ParamBase {
  const lite_metal::Tensor* x{};
  const lite_metal::Tensor* wi{};
  const lite_metal::Tensor* wh{};
  int num_input;
  int num_hidden;

  lite_metal::Tensor* out{};
  lite_metal::Tensor* tmp_buffer{};
  lite_metal::Tensor* idx_sorted_by_width{};
  lite_metal::Tensor* layout_input{};

#ifdef LITE_WITH_XPU
  bool __xpu__float_to_fix{false};   // Is wi/wh already converted to int16/int8
  std::vector<float> __xpu__wi_max;  // Abs max in wi
  std::vector<float> __xpu__wh_max;  // Abs max in wh
#endif
};

struct SplitLodTensorParam : ParamBase {
  const lite_metal::Tensor* x{};
  const lite_metal::Tensor* mask{};
  lite_metal::Tensor* out_true{};
  lite_metal::Tensor* out_false{};
  int level{};
};

struct MergeLodTensorParam : ParamBase {
  const lite_metal::Tensor* x{};
  const lite_metal::Tensor* mask{};
  const lite_metal::Tensor* in_true{};
  const lite_metal::Tensor* in_false{};
  lite_metal::Tensor* out{};
  int level{};
};

struct ConditionalBlockParam : ParamBase {
  const lite_metal::Tensor* cond{};
  std::vector<lite_metal::Tensor*> inputs{};
  std::vector<lite_metal::Tensor*> outs{};
  int block_idx{-1};
  std::shared_ptr<const cpp::ProgramDesc> program_desc{nullptr};
  Scope* exec_scope{nullptr};
  bool is_scalar_condition{};
};

struct CollectFpnProposalsParam : ParamBase {
  std::vector<lite_metal::Tensor*> multi_level_rois{};
  std::vector<lite_metal::Tensor*> multi_level_scores{};
  std::vector<lite_metal::Tensor*> multi_rois_num{};
  lite_metal::Tensor* rois_num{};
  lite_metal::Tensor* fpn_rois{};
  int post_nms_topN{};
};

struct DistributeFpnProposalsParam : ParamBase {
  const lite_metal::Tensor* fpn_rois{};
  const lite_metal::Tensor* rois_num{};
  std::vector<lite_metal::Tensor*> multi_fpn_rois{};
  std::vector<lite_metal::Tensor*> multi_rois_num{};
  lite_metal::Tensor* restore_index{};
  int min_level{};
  int max_level{};
  int refer_level{};
  int refer_scale{};
};

/// --------------------- instance_norm operators --------------------
struct InstanceNormParam : ParamBase {
  lite_metal::Tensor* x{};
  lite_metal::Tensor* out{};
  lite_metal::Tensor* bias{};
  lite_metal::Tensor* scale{};
  lite_metal::Tensor* saved_mean{};
  lite_metal::Tensor* saved_variance{};
  float epsilon;
  bool fuse_relu{false};
  std::string activation_type{""};
  float alpha{6.f};
};
/// --------------------- group_norm operators --------------------
struct GroupNormParam : ParamBase {
  lite_metal::Tensor* x{};
  lite_metal::Tensor* out{};
  lite_metal::Tensor* bias{};
  lite_metal::Tensor* scale{};
  lite_metal::Tensor* saved_mean{};
  lite_metal::Tensor* saved_variance{};
  std::string data_layout_str{"NCHW"};
  float epsilon;
  int groups;
  int channels;
};

/// --------------------- grid sampler operators --------------------
struct GridSamplerParam : ParamBase {
  const lite_metal::Tensor* x{nullptr};
  const lite_metal::Tensor* grid{nullptr};
  lite_metal::Tensor* out{nullptr};
  bool align_corners{true};
  std::string padding_mode{"zeros"};
  std::string mode{"bilinear"};
};

struct LstmParam : ParamBase {
  lite_metal::Tensor* Input{};
  lite_metal::Tensor* Weight{};
  lite_metal::Tensor* Bias{};
  lite_metal::Tensor* Hidden{};
  lite_metal::Tensor* Cell{};
  lite_metal::Tensor* BatchGate{};
  lite_metal::Tensor* BatchCellPreAct{};
  lite_metal::Tensor* H0{nullptr};
  lite_metal::Tensor* C0{nullptr};
  bool use_peepholes;
  bool is_reverse;
  lite_metal_api::ActivationType gate_activation;
  lite_metal_api::ActivationType cell_activation;
  lite_metal_api::ActivationType candidate_activation;
  // for int8
  WITH_INT8_CONFIG
};

struct CrfDecodingParam : ParamBase {
  lite_metal::Tensor* emission{};
  lite_metal::Tensor* transition{};
  lite_metal::Tensor* label{};
  lite_metal::Tensor* length{};
  lite_metal::Tensor* viterbi_path{};
};

struct CtcAlignParam : ParamBase {
  lite_metal::Tensor* input{};
  lite_metal::Tensor* input_length{};
  lite_metal::Tensor* output{};
  lite_metal::Tensor* output_length{};
  int blank{0};
  bool merge_repeated{true};
  int padding_value{0};
};

struct XPUResNet50Param : ParamBase {
  lite_metal::Tensor* input{};
  std::vector<lite_metal::Tensor*> filter;
  std::vector<lite_metal::Tensor*> bias;
  std::vector<lite_metal::Tensor*> max_filter;
  lite_metal::Tensor* output{};
};

struct XPUSoftmaxTopkParam : ParamBase {
  const lite_metal::Tensor* x{};
  lite_metal::Tensor* output{};
  lite_metal::Tensor* indices{};
  int axis{-1};
  int K{1};
};

struct XPUBlockFuseParam : ParamBase {
  const lite_metal::Tensor* input{nullptr};
  const lite_metal::Tensor* filter{nullptr};
  const lite_metal::Tensor* bias{nullptr};
  const lite_metal::Tensor* branch{nullptr};
  const lite_metal::Tensor* input_max{nullptr};
  lite_metal::Tensor* output{nullptr};
  lite_metal::Tensor* output_max{nullptr};
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
};

struct XPUMultiEncoderParam : ParamBase {
  lite_metal::Tensor* input{};
  std::vector<lite_metal::Tensor*> fc_weight;
  std::vector<lite_metal::Tensor*> fc_bias;
  std::vector<lite_metal::Tensor*> ln_scale;
  std::vector<lite_metal::Tensor*> ln_bias;
  lite_metal::Tensor* fc_weight_max{};
  const lite_metal::Tensor* mask{nullptr};
  const lite_metal::Tensor* SeqLod{nullptr};
  const lite_metal::Tensor* PadSeqLen{nullptr};
  lite_metal::Tensor* output{nullptr};

  std::vector<int> slice_axes{};
  std::vector<int> slice_starts{};
  std::vector<int> slice_ends{};
  std::vector<int> slice_decrease_axis{};
  int n_layers{};
  int head_num{};
  int size_per_head{};
  std::string act_type{};
  std::string precision{};
  bool enable_qkv_fusion{false};
  bool norm_before{false};
  bool adaptive_seqlen{false};
};

struct XPUEmbeddingWithEltwiseAddParam : ParamBase {
  std::vector<lite_metal::Tensor*> Ids;
  std::vector<lite_metal::Tensor*> Tables;
  const lite_metal::Tensor* Mask{nullptr};
  lite_metal::Tensor* SeqLod{nullptr};
  lite_metal::Tensor* PadSeqLen{nullptr};
  lite_metal::Tensor* Out{nullptr};
  int64_t padding_idx{-1};
};

struct XPUFcParam : ParamBase {
  const lite_metal::Tensor* input{nullptr};
  const lite_metal::Tensor* w{nullptr};
  const lite_metal::Tensor* bias{nullptr};
  const lite_metal::Tensor* input_max{nullptr};
  lite_metal::Tensor* output{nullptr};
  lite_metal::Tensor* output_max{nullptr};
  lite_metal::DDim in_mat_dims;

  int act_type;
  float act_param;
  std::string precision{};
  bool has_bias{false};
  int in_num_col_dims{1};
};

struct XPUResNetCbamParam : ParamBase {
  lite_metal::Tensor* input{};
  std::vector<lite_metal::Tensor*> filter;
  std::vector<lite_metal::Tensor*> bias;
  std::vector<lite_metal::Tensor*> max_filter;
  lite_metal::Tensor* output{};

  float pool_p{1.0f};
};

struct XPUMmdnnSearchAttentionParam : ParamBase {
  lite_metal::Tensor* X{};
  lite_metal::Tensor* W{};
  lite_metal::Tensor* b{};
  lite_metal::Tensor* Out{};

  float W_max{0.0f};
  int pad_id{0};
  float alpha0{1.0f};
  float alpha1{1.0f};
  float mask{1.0f};
};

struct XPUMmdnnBidEmbGrnnAttParam : ParamBase {
  lite_metal::Tensor* id0{};
  lite_metal::Tensor* id1{};
  lite_metal::Tensor* emb_tbl{};
  lite_metal::Tensor* grnn_fw_wh{};
  lite_metal::Tensor* grnn_fw_wi{};
  lite_metal::Tensor* grnn_rv_wh{};
  lite_metal::Tensor* grnn_rv_wi{};
  lite_metal::Tensor* att_fc_w{};
  lite_metal::Tensor* att_fc_b{};

  std::vector<float> grnn_fw_wh_maxs;
  std::vector<float> grnn_fw_wi_maxs;
  std::vector<float> grnn_rv_wh_maxs;
  std::vector<float> grnn_rv_wi_maxs;
  float att_fc_w_max{0.0f};

  lite_metal::Tensor* grnn_fw_pool_out{};
  lite_metal::Tensor* grnn_rv_pool_out{};
  lite_metal::Tensor* att_pool_out{};
  lite_metal::Tensor* concat_3in1_out{};
  lite_metal::Tensor* emb_fw_out{};
};

struct XPUMmdnnBidEmbGrnnAttParam2 : ParamBase {
  lite_metal::Tensor* id0{};
  lite_metal::Tensor* id1{};
  lite_metal::Tensor* emb_tbl{};
  lite_metal::Tensor* grnn_fw_wh{};
  lite_metal::Tensor* grnn_fw_wi{};
  lite_metal::Tensor* grnn_rv_wh{};
  lite_metal::Tensor* grnn_rv_wi{};
  lite_metal::Tensor* att_fc_w{};
  lite_metal::Tensor* att_fc_b{};

  std::vector<float> grnn_fw_wh_maxs;
  std::vector<float> grnn_fw_wi_maxs;
  std::vector<float> grnn_rv_wh_maxs;
  std::vector<float> grnn_rv_wi_maxs;
  float att_fc_w_max{0.0f};

  lite_metal::Tensor* emb0_out{};
  lite_metal::Tensor* grnn_fw_pool_out{};
  lite_metal::Tensor* grnn_rv_pool_out{};
  lite_metal::Tensor* att_pool_out{};
  lite_metal::Tensor* concat_3in1_out{};
  lite_metal::Tensor* emb_fw_out{};
};

struct XPUMmdnnBidEmbAttParam : ParamBase {
  lite_metal::Tensor* id0{};
  lite_metal::Tensor* id1{};
  lite_metal::Tensor* emb_tbl{};
  lite_metal::Tensor* att_fc_w{};
  lite_metal::Tensor* att_fc_b{};

  float att_fc_w_max{0.0f};

  lite_metal::Tensor* att_pool_out{};
  lite_metal::Tensor* emb_fw_out{};
};

struct XPUMmdnnMatchConvTopkParam : ParamBase {
  lite_metal::Tensor* input_x{};
  lite_metal::Tensor* input_y{};
  lite_metal::Tensor* input_w{};
  lite_metal::Tensor* conv_w{};

  float input_w_max{0.0f};
  float conv_w_max{0.0f};
  std::vector<int> topks;
  int output_channel{0};
  int channel_num{0};
  int dim_t{0};

  lite_metal::Tensor* topk_out{};
};

struct XPUMmdnnMergeAllParam : ParamBase {
  std::vector<lite_metal::Tensor*> concat_7in1_x;
  std::vector<lite_metal::Tensor*> concat_topk_x;
  lite_metal::Tensor* grnn_fw_wh{};
  lite_metal::Tensor* grnn_fw_wi{};
  lite_metal::Tensor* grnn_rv_wh{};
  lite_metal::Tensor* grnn_rv_wi{};
  lite_metal::Tensor* fc0_w{};
  lite_metal::Tensor* fc0_b{};
  lite_metal::Tensor* fc1_w{};
  lite_metal::Tensor* fc1_b{};
  lite_metal::Tensor* fc2_w{};
  lite_metal::Tensor* fc2_b{};

  std::vector<float> grnn_fw_wh_maxs;
  std::vector<float> grnn_fw_wi_maxs;
  std::vector<float> grnn_rv_wh_maxs;
  std::vector<float> grnn_rv_wi_maxs;
  float fc0_w_max{0.0f};
  float fc1_w_max{0.0f};
  float fc2_w_max{0.0f};

  lite_metal::Tensor* out{};
};

struct XPUSfaHeadParam : ParamBase {
  lite_metal::Tensor* input{nullptr};
  lite_metal::Tensor* output{nullptr};

  std::string op_type{""};
};

struct XPUGenerateSequenceParam : ParamBase {
  const lite_metal::Tensor* input{nullptr};
  lite_metal::Tensor* output{nullptr};

  int axis{-1};
  bool flatten{false};
  float value{0.f};
  int dtype{-1};
};

struct XPULogitParam : ParamBase {
  const lite_metal::Tensor* input{nullptr};
  lite_metal::Tensor* output{nullptr};
  float eps{1e-7f};
};

struct XPUConvPixelShuffleFuseParam : ParamBase {
  const lite_metal::Tensor* input{nullptr};
  const lite_metal::Tensor* filter_0{nullptr};
  const lite_metal::Tensor* filter_1{nullptr};
  const lite_metal::Tensor* bias_0{nullptr};
  const lite_metal::Tensor* bias_1{nullptr};
  const lite_metal::Tensor* input_max{nullptr};
  lite_metal::Tensor* output{nullptr};
  lite_metal::Tensor* output_max{nullptr};
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

// For DeformableConvolution op
struct DeformableConvParam : ParamBase {
  lite_metal::Tensor* x{};
  lite_metal::Tensor* offset{};
  lite_metal::Tensor* mask{};
  lite_metal::Tensor* output{};
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
      output_tensor_ptrs_cache_.reset(new std::vector<lite_metal::Tensor*>({output}));
    }
    return output_tensor_ptrs_cache_.get();
  }
};

struct PixelShuffleParam : ParamBase {
  lite_metal::Tensor* x{nullptr};
  lite_metal::Tensor* output{nullptr};
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
  const lite_metal::Tensor* input{nullptr};
  lite_metal::Tensor* output{nullptr};
};

struct WhereParam : ParamBase {
  const lite_metal::Tensor* x{nullptr};
  const lite_metal::Tensor* y{nullptr};
  const lite_metal::Tensor* condition{nullptr};
  lite_metal::Tensor* out{nullptr};
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
  const lite_metal::Tensor* in{};
  lite_metal::Tensor* out{};
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
  const lite_metal::Tensor* X{};
  const lite_metal::Tensor* depth_tensor{nullptr};
  lite_metal::Tensor* Out{};
  int depth;
  int dtype;
  bool allow_out_of_range;
};

struct TrigonometricParam : ParamBase {
  lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
};

using SinParam = TrigonometricParam;
using CosParam = TrigonometricParam;

struct FlattenContiguousRangeParam : ParamBase {
  lite_metal::Tensor* x{};
  lite_metal::Tensor* out{};
  lite_metal::Tensor* xshape;
  int start_axis;
  int stop_axis;
};

struct LoDArrayLengthParam : ParamBase {
  std::vector<lite_metal::Tensor>* x{};
  lite_metal::Tensor* out{};
};

struct SelectInputParam : ParamBase {
  std::vector<lite_metal::Tensor*> X{};
  lite_metal::Tensor* Mask{};
  lite_metal::Tensor* Out{};
};

struct TensorArrayToTensorParam : ParamBase {
  std::vector<lite_metal::Tensor>* X{};
  lite_metal::Tensor* Out{};
  lite_metal::Tensor* OutIndex{};
  int axis{0};
  bool use_stack{false};
};

struct RnnParam : ParamBase {
  lite_metal::Tensor* Input;
  std::vector<lite_metal::Tensor*> PreState;
  std::vector<lite_metal::Tensor*> WeightList;
  const lite_metal::Tensor* SequenceLength{nullptr};
  lite_metal::Tensor* DropoutState;
  lite_metal::Tensor* Reserve;
  lite_metal::Tensor* Out;
  std::vector<lite_metal::Tensor*> State;
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
  lite_metal::Tensor* Input{};
  lite_metal::Tensor* Out{};
  std::vector<int> starts{};
  std::vector<int> ends{};
  std::vector<int> strides{};
  std::vector<int> axes{};
  std::vector<int> infer_flags{};
  std::vector<int> decrease_axis{};
  std::vector<lite_metal::Tensor*> StartsTensorList{};
  std::vector<lite_metal::Tensor*> EndsTensorList{};
  std::vector<lite_metal::Tensor*> StridesTensorList{};
  bool tensor_input{false};
  lite_metal::Tensor* EndsTensor{nullptr};
  lite_metal::Tensor* StartsTensor{nullptr};
  lite_metal::Tensor* StridesTensor{nullptr};
};

struct TileParam : ParamBase {
  lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
  std::vector<int> repeat_times{};
  lite_metal::Tensor* RepeatTimes{};
  std::vector<lite_metal::Tensor*> repeat_times_tensor{};
};

struct ScatterNdAddParam : ParamBase {
  const lite_metal::Tensor* x{};
  lite_metal::Tensor* indexs{};
  lite_metal::Tensor* updates{};
  lite_metal::Tensor* output{};
};

struct CumsumParam : ParamBase {
  const lite_metal::Tensor* X{nullptr};
  lite_metal::Tensor* Out{nullptr};

  int axis{-1};
  bool flatten{false};
  bool exclusive{false};
  bool reverse{false};
};

struct PolygonBoxTransformParam : ParamBase {
  const lite_metal::Tensor* input{nullptr};
  lite_metal::Tensor* output{nullptr};
};

struct SumParam : ParamBase {
  std::vector<lite_metal::Tensor*> X{};
  lite_metal::Tensor* Out{};
  int inplace{0};
};

struct PNormParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};

  float porder{2.f};
  int axis{-1};
  float epsilon{1.0e-12f};
  bool keepdim{false};
  bool asvector{false};
};

struct LinspaceParam : ParamBase {
  const lite_metal::Tensor* Start{};
  const lite_metal::Tensor* Stop{};
  const lite_metal::Tensor* Num{};
  lite_metal::Tensor* Out{};
  int dtype{};
};

struct RoiPerspectiveTransformParam : ParamBase {
  const lite_metal::Tensor* x{nullptr};
  const lite_metal::Tensor* rois{nullptr};
  lite_metal::Tensor* out{nullptr};
  lite_metal::Tensor* mask{nullptr};
  lite_metal::Tensor* transfor_matrix{nullptr};
  lite_metal::Tensor* out2in_idx{nullptr};
  lite_metal::Tensor* out2in_weight{nullptr};
  float spatial_scale{1.f};
  int transformed_height{1};
  int transformed_width{1};
};

struct CorrelationParam : ParamBase {
  const lite_metal::Tensor* input1{nullptr};
  const lite_metal::Tensor* input2{nullptr};
  lite_metal::Tensor* output{nullptr};
  int pad_size;
  int kernel_size;
  int max_displacement;
  int stride1;
  int stride2;
  int corr_type_multiply{1};
};

struct ArgsortParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};
  lite_metal::Tensor* Indices{};

  int axis{-1};
  bool descending{false};
};

struct FlipParam : ParamBase {
  const lite_metal::Tensor* X{};
  lite_metal::Tensor* Out{};

  std::vector<int> axis;
};

struct WriteBackParam : ParamBase {
  const lite_metal::Tensor* x{};
  lite_metal::Tensor* y{};
};

}  // namespace operators
}  // namespace lite
}  // namespace paddle
