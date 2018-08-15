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

#include <string>
#include <vector>
#include "common/log.h"
#include "common/type_define.h"
#include "framework/lod_tensor.h"
#include "framework/scope.h"
#include "framework/tensor.h"
#include "framework/variable.h"
#ifdef PADDLE_MOBILE_FPGA
#include "fpga/api/fpga_api.h"
#endif

namespace paddle_mobile {
namespace operators {

using framework::Attribute;
using framework::AttributeMap;
using framework::LoDTensor;
using framework::Scope;
using framework::Tensor;
using std::string;
using std::vector;

class OpParam {
 protected:
  template <typename T>
  static T *InputFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("Input", inputs, scope);
  }

  template <typename T>
  static T *InputXFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("X", inputs, scope);
  }

  template <typename T>
  static T *InputYFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("Y", inputs, scope);
  }

  template <typename T>
  static T *InputZFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("Z", inputs, scope);
  }

  template <typename T>
  static T *InputBiasFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("Bias", inputs, scope);
  }
  template <typename T>
  static T *InputVarianceFrom(const VariableNameMap &inputs,
                              const Scope &scope) {
    return GetVarValue<T>("Variance", inputs, scope);
  }
  template <typename T>
  static T *InputMeanFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("Mean", inputs, scope);
  }
  template <typename T>
  static T *InputScaleFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("Scale", inputs, scope);
  }
  template <typename T>
  static T *InputImageFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("Image", inputs, scope);
  }
  template <typename T>
  static T *InputPriorBoxFrom(const VariableNameMap &inputs,
                              const Scope &scope) {
    return GetVarValue<T>("PriorBox", inputs, scope);
  }
  template <typename T>
  static T *InputPriorBoxVarFrom(const VariableNameMap &inputs,
                                 const Scope &scope) {
    return GetVarValue<T>("PriorBoxVar", inputs, scope);
  }
  // LoDTensor but now use Tensor
  template <typename T>
  static T *InputTargetBoxFrom(const VariableNameMap &inputs,
                               const Scope &scope) {
    return GetVarValue<T>("TargetBox", inputs, scope);
  }

  template <typename T>
  static T *InputBBoxesFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("BBoxes", inputs, scope);
  }

  template <typename T>
  static T *InputScoresFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("Scores", inputs, scope);
  }

  template <typename T>
  static T *InputShapeFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("Shape", inputs, scope);
  }

  template <typename T>
  static vector<T *> InputMultiFrom(const VariableNameMap &inputs,
                                    const Scope &scope) {
    return GetMultiVarValue<T>("X", inputs, scope);
  }

  template <typename T>
  static T *OutputFrom(const VariableNameMap &outputs, const Scope &scope) {
    return GetVarValue<T>("Output", outputs, scope);
  }

  template <typename T>
  static T *OutFrom(const VariableNameMap &outputs, const Scope &scope) {
    return GetVarValue<T>("Out", outputs, scope);
  }

  template <typename T>
  static T *OutputYFrom(const VariableNameMap &outputs, const Scope &scope) {
    return GetVarValue<T>("Y", outputs, scope);
  }

  template <typename T>
  static T *OutputBoxesFrom(const VariableNameMap &outputs,
                            const Scope &scope) {
    return GetVarValue<T>("Boxes", outputs, scope);
  }

  template <typename T>
  static T *OutputBoxFrom(const VariableNameMap &outputs, const Scope &scope) {
    return GetVarValue<T>("OutputBox", outputs, scope);
  }

  template <typename T>
  static T *OutputVariancesFrom(const VariableNameMap &outputs,
                                const Scope &scope) {
    return GetVarValue<T>("Variances", outputs, scope);
  }

  template <typename T>
  static T *MidOutFrom(const VariableNameMap &outputs, const Scope &scope) {
    return GetVarValue<T>("MidOut", outputs, scope);
  }

  template <typename T>
  static T *FilterFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("Filter", inputs, scope);
  }

  template <typename T>
  static const T GetAttr(const string &key, const AttributeMap &map) {
    return ((Attribute)map.at(key)).Get<T>();
  }

  template <typename T>
  static T *GetVarValue(const string &key, const VariableNameMap &var_map,
                        const Scope &scope) {
    PADDLE_MOBILE_ENFORCE(var_map.count(key) > 0,
                          "%s is not contained in var_map", key.c_str())
    auto var_vec = var_map.at(key);
    if (!var_vec.empty()) {
      auto var = scope.FindVar(var_vec[0]);
      return var->GetMutable<T>();
    } else {
      return nullptr;
    }
  }

  template <typename T>
  static vector<T *> GetMultiVarValue(const string &key,
                                      const VariableNameMap &var_map,
                                      const Scope &scope) {
    auto var_vecs = var_map.at(key);
    assert(var_vecs.size() > 1);
    vector<T *> var_res;
    for (auto &var_vec : var_vecs) {
      auto var = scope.FindVar(var_vec);
      var_res.push_back(var->GetMutable<T>());
    }
    return var_res;
  }
};

#ifdef CONV_OP
class ConvParam : OpParam {
 public:
  ConvParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
            const AttributeMap &attrs, const Scope &scope) {
    filter_ = FilterFrom<LoDTensor>(inputs, scope);
    input_ = InputFrom<LoDTensor>(inputs, scope);
    output_ = OutputFrom<LoDTensor>(outputs, scope);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
  }

  const Tensor *Input() const { return input_; }

  Tensor *Filter() const { return filter_; }

  Tensor *Output() const { return output_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

 private:
  Tensor *input_;
  Tensor *output_;
  Tensor *filter_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;
  int groups;
};

Print &operator<<(Print &printer, const ConvParam &conv_param);
#endif

class ElementwiseAddParam : OpParam {
 public:
  ElementwiseAddParam(const VariableNameMap &inputs,
                      const VariableNameMap &outputs, const AttributeMap &attrs,
                      const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    input_y_ = InputYFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
    axis_ = GetAttr<int>("axis", attrs);
  }

  const Tensor *InputX() const { return input_x_; }

  const Tensor *InputY() const { return input_y_; }

  Tensor *Out() const { return out_; }

  const int &Axis() const { return axis_; }

 private:
  Tensor *input_x_;
  Tensor *input_y_;
  Tensor *out_;
  int axis_;
#ifdef PADDLE_MOBILE_FPGA

 private:
  fpga::EWAddArgs fpga_EW_add_args;

 public:
  const fpga::EWAddArgs &FpgaArgs() const { return fpga_EW_add_args; }
  void SetFpgaArgs(const fpga::EWAddArgs &args) { fpga_EW_add_args = args; }
#endif
};

#ifdef FUSION_ELEMENTWISEADDRELU_OP
using ElementwiseAddReluParam = ElementwiseAddParam;
#endif

#ifdef MUL_OP
class MulParam : OpParam {
 public:
  MulParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
           const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    input_y_ = InputYFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
    x_num_col_dims_ = GetAttr<int>("x_num_col_dims", attrs);
    y_num_col_dims_ = GetAttr<int>("y_num_col_dims", attrs);
  }

  const Tensor *InputX() const { return input_x_; }

  const Tensor *InputY() const { return input_y_; }

  Tensor *Out() const { return out_; }

  const int &XNumColDims() const { return x_num_col_dims_; }

  const int &YNumColDims() const { return y_num_col_dims_; }

 private:
  Tensor *input_x_;
  Tensor *input_y_;
  Tensor *out_;
  int x_num_col_dims_;
  int y_num_col_dims_;
};
#endif

#ifdef CONCAT_OP
class ConcatParam : public OpParam {
 public:
  ConcatParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
              const AttributeMap &attrs, const Scope &scope) {
    inputs_ = InputMultiFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
    axis_ = GetAttr<int>("axis", attrs);
  }

  vector<LoDTensor *> Inputs() const { return inputs_; }

  Tensor *Out() const { return out_; }

  const int &Axis() const { return axis_; }

 private:
  vector<LoDTensor *> inputs_;
  Tensor *out_;
  int axis_;
};
#endif

#ifdef LRN_OP
class LrnParam : public OpParam {
 public:
  LrnParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
           const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
    mid_out_ = MidOutFrom<LoDTensor>(outputs, scope);
    n_ = GetAttr<int>("n", attrs);
    alpha_ = GetAttr<float>("alpha", attrs);
    beta_ = GetAttr<float>("beta", attrs);
    k_ = GetAttr<float>("k", attrs);
    data_format_ = GetAttr<string>("data_format", attrs);
  }

  const Tensor *InputX() const { return input_x_; }

  Tensor *Out() const { return out_; }

  Tensor *MidOut() const { return mid_out_; }

  const int &N() const { return n_; }

  const float &Alpha() const { return alpha_; }

  const float &Beta() const { return beta_; }

  const float &K() const { return k_; }

  const string &DataFormat() const { return data_format_; }

 private:
  Tensor *input_x_;
  Tensor *out_;
  Tensor *mid_out_;
  int n_;
  float alpha_;
  float beta_;
  float k_;
  string data_format_;
};
#endif

#ifdef BATCHNORM_OP
class BatchNormParam : OpParam {
 public:
  BatchNormParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
                 const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    output_y_ = OutputYFrom<LoDTensor>(outputs, scope);
    input_bias_ = InputBiasFrom<LoDTensor>(inputs, scope);
    input_mean_ = InputMeanFrom<LoDTensor>(inputs, scope);
    input_scale_ = InputScaleFrom<LoDTensor>(inputs, scope);
    input_variance_ = InputVarianceFrom<LoDTensor>(inputs, scope);
    epsilon_ = GetAttr<float>("epsilon", attrs);
    momentum_ = GetAttr<float>("momentum", attrs);
    //    is_test_ = GetAttr<bool>("is_test", attrs);
  }

  const Tensor *InputX() const { return input_x_; }

  Tensor *OutputY() const { return output_y_; }

  const Tensor *InputBias() const { return input_bias_; }

  const Tensor *InputMean() const { return input_mean_; }

  const Tensor *InputScale() const { return input_scale_; }

  const Tensor *InputVariance() const { return input_variance_; }

  const float &Epsilon() const { return epsilon_; }

  const float &Momentum() const { return momentum_; }

  const bool &IsTest() const { return is_test_; }

  const string &DataFormat() const { return data_format_; }

 private:
  Tensor *input_x_;
  Tensor *output_y_;
  Tensor *input_bias_;
  Tensor *input_mean_;
  Tensor *input_scale_;
  Tensor *input_variance_;
  float epsilon_;
  float momentum_;
  bool is_test_;
  string data_format_;
};
#endif

#ifdef POOL_OP
class PoolParam : public OpParam {
 public:
  PoolParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
            const AttributeMap &attrs, const Scope &scope) {
    input_ = InputXFrom<LoDTensor>(inputs, scope);

    output_ = OutFrom<LoDTensor>(outputs, scope);
    pooling_type_ = GetAttr<string>("pooling_type", attrs);
    ksize_ = GetAttr<vector<int>>("ksize", attrs);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    ceil_mode_ = GetAttr<bool>("ceil_mode", attrs);
    global_pooling_ = GetAttr<bool>("global_pooling", attrs);
  }

  const Tensor *Input() const { return input_; }

  Tensor *Output() const { return output_; }

  const string &PoolingType() const { return pooling_type_; }

  const vector<int> &Ksize() const { return ksize_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  bool isCeilMode() const { return ceil_mode_; }

  bool isGlobalPooling() const { return global_pooling_; }

 private:
  Tensor *input_;
  Tensor *output_;
  string pooling_type_;
  vector<int> ksize_;
  vector<int> strides_;
  vector<int> paddings_;
  bool ceil_mode_;
  bool global_pooling_ = false;
#ifdef PADDLE_MOBILE_FPGA

 private:
  fpga::PoolingArgs fpga_pool_args;

 public:
  const fpga::PoolingArgs &FpgaArgs() const { return fpga_pool_args; }
  void SetFpgaArgs(const fpga::PoolingArgs &args) { fpga_pool_args = args; }
#endif
};
#endif

#ifdef PRIORBOX_OP
class PriorBoxParam : public OpParam {
 public:
  PriorBoxParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
                const AttributeMap &attrs, const Scope &scope) {
    input_ = InputFrom<LoDTensor>(inputs, scope);
    input_image_ = InputImageFrom<LoDTensor>(inputs, scope);
    output_boxes_ = OutputBoxesFrom<LoDTensor>(outputs, scope);
    output_variances_ = OutputVariancesFrom<LoDTensor>(outputs, scope);
    min_sizes_ = GetAttr<vector<float>>("min_sizes", attrs);
    max_sizes_ = GetAttr<vector<float>>("max_sizes", attrs);
    aspect_ratios_ = GetAttr<vector<float>>("aspect_ratios", attrs);
    variances_ = GetAttr<vector<float>>("variances", attrs);
    flip_ = GetAttr<bool>("flip", attrs);
    clip_ = GetAttr<bool>("clip", attrs);
    step_w_ = GetAttr<float>("step_w", attrs);
    step_h_ = GetAttr<float>("step_h", attrs);
    offset_ = GetAttr<float>("offset", attrs);
  }
  const Tensor *Input() const { return input_; }

  const Tensor *InputImage() const { return input_image_; }

  Tensor *OutputBoxes() const { return output_boxes_; }

  Tensor *OutputVariances() const { return output_variances_; }

  const vector<float> &MinSizes() const { return min_sizes_; }

  const vector<float> &MaxSizes() const { return max_sizes_; }

  const vector<float> &AspectRatios() const { return aspect_ratios_; }

  const vector<float> &Variances() const { return variances_; }

  const bool &Flip() const { return flip_; }

  const bool &Clip() const { return clip_; }

  const float &StepW() const { return step_w_; }

  const float &StepH() const { return step_h_; }

  const float &Offset() const { return offset_; }

 private:
  Tensor *input_;
  Tensor *input_image_;
  Tensor *output_boxes_;
  Tensor *output_variances_;
  vector<float> min_sizes_;
  vector<float> max_sizes_;
  vector<float> aspect_ratios_;
  vector<float> variances_;
  bool flip_;
  bool clip_;
  float step_w_;
  float step_h_;
  float offset_;
};
#endif

#ifdef BOXCODER_OP
class BoxCoderParam : public OpParam {
 public:
  BoxCoderParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
                const AttributeMap &attrs, const Scope &scope) {
    input_priorbox_ = InputPriorBoxFrom<LoDTensor>(inputs, scope);
    input_priorboxvar_ = InputPriorBoxVarFrom<LoDTensor>(inputs, scope);
    input_targetbox_ = InputTargetBoxFrom<LoDTensor>(inputs, scope);
    output_box_ = OutputBoxFrom<LoDTensor>(outputs, scope);
    code_type_ = GetAttr<std::string>("code_type", attrs);
  }
  const Tensor *InputPriorBox() const { return input_priorbox_; }

  const Tensor *InputPriorBoxVar() const { return input_priorboxvar_; }

  const Tensor *InputTargetBox() const { return input_targetbox_; }

  Tensor *OutputBox() const { return output_box_; }

  const std::string &CodeType() const { return code_type_; }

 private:
  Tensor *input_priorbox_;
  Tensor *input_priorboxvar_;
  Tensor *input_targetbox_;
  Tensor *output_box_;
  std::string code_type_;
};
#endif

#ifdef SOFTMAX_OP
class SoftmaxParam : public OpParam {
 public:
  SoftmaxParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
               const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
  }
  const Tensor *InputX() const { return input_x_; }
  Tensor *Out() const { return out_; }

 private:
  Tensor *input_x_;
  Tensor *out_;
};
#endif

#ifdef SIGMOID_OP
class SigmoidParam : public OpParam {
 public:
  SigmoidParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
               const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
  }
  const Tensor *InputX() const { return input_x_; }
  Tensor *Out() const { return out_; }

 private:
  Tensor *input_x_;
  Tensor *out_;
};
#endif

#ifdef MULTICLASSNMS_OP
class MultiClassNMSParam : public OpParam {
 public:
  MultiClassNMSParam(const VariableNameMap &inputs,
                     const VariableNameMap &outputs, const AttributeMap &attrs,
                     const Scope &scope) {
    input_bboxes_ = InputBBoxesFrom<LoDTensor>(inputs, scope);
    input_scores_ = InputScoresFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
    background_label_ = GetAttr<int>("background_label", attrs);
    nms_top_k_ = GetAttr<int>("nms_top_k", attrs);
    keep_top_k_ = GetAttr<int>("keep_top_k", attrs);
    nms_threshold_ = GetAttr<float>("nms_threshold", attrs);
    nms_eta_ = GetAttr<float>("nms_eta", attrs);
    score_threshold_ = GetAttr<float>("score_threshold", attrs);
  }

  const Tensor *InputBBoxes() const { return input_bboxes_; }

  const Tensor *InputScores() const { return input_scores_; }

  Tensor *Out() const { return out_; }

  const int &BackGroundLabel() const { return background_label_; }

  const int &NMSTopK() const { return nms_top_k_; }

  const int &KeepTopK() const { return keep_top_k_; }

  const float &NMSThreshold() const { return nms_threshold_; }

  const float &NMSEta() const { return nms_eta_; }

  const float &ScoreThreshold() const { return score_threshold_; }

 private:
  Tensor *input_bboxes_;
  Tensor *input_scores_;
  Tensor *out_;
  int background_label_;
  int nms_top_k_;
  int keep_top_k_;
  float nms_threshold_;
  float nms_eta_;
  float score_threshold_;
};
#endif

class FeedParam : public OpParam {
 public:
  FeedParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
            const AttributeMap &attrs, Scope *scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, *scope);
    out_ = OutFrom<LoDTensor>(outputs, *scope);
    auto var = scope->Var("batch_size");
    batch_size = var->GetValue<int>();
  }
  const Tensor *InputX() const { return input_x_; }
  Tensor *Out() const { return out_; }
  const int BatchSize() const { return batch_size; }

 private:
  Tensor *input_x_;
  Tensor *out_;
  int batch_size;

#ifdef PADDLE_MOBILE_FPGA

 private:
  fpga::BypassArgs fpga_bypass_args;

 public:
  const fpga::BypassArgs &FpgaArgs() const { return fpga_bypass_args; }
  void SetFpgaArgs(const fpga::BypassArgs &args) { fpga_bypass_args = args; }
#endif
};

class FetchParam : public OpParam {
 public:
  FetchParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
             const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
  }
  const Tensor *InputX() const { return input_x_; }
  Tensor *Out() const { return out_; }

 private:
  Tensor *input_x_;
  Tensor *out_;
};

#ifdef TRANSPOSE_OP
class TransposeParam : public OpParam {
 public:
  TransposeParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
                 const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
    axis_ = GetAttr<vector<int>>("axis", attrs);
  }

  const Tensor *InputX() const { return input_x_; }

  Tensor *Out() const { return out_; }

  const vector<int> &Axis() const { return axis_; }

 private:
  Tensor *input_x_;
  Tensor *out_;
  vector<int> axis_;
};
#endif

#ifdef RESHAPE_OP
class ReshapeParam : public OpParam {
 public:
  ReshapeParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
               const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    input_shape_ = InputShapeFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
    shape_ = GetAttr<vector<int>>("shape", attrs);
    inplace_ = GetAttr<bool>("inplace", attrs);
  }

  const Tensor *InputX() const { return input_x_; }

  const Tensor *InputShape() const { return input_shape_; }

  Tensor *Out() const { return out_; }

  const vector<int> &Shape() const { return shape_; }

  const bool &Inplace() const { return inplace_; }

 private:
  Tensor *input_x_;
  Tensor *input_shape_;
  Tensor *out_;
  vector<int> shape_;
  bool inplace_;
};
#endif

#ifdef SCALE_OP
class ScaleParam : public OpParam {
 public:
  ScaleParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
             const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    input_bias_ = InputBiasFrom<framework::LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
    inplace_ = GetAttr<bool>("inplace", attrs);
    has_bias_ = GetAttr<bool>("has_bias", attrs);
    scales_ = GetAttr<vector<float>>("scales", attrs);
    biases_ = GetAttr<vector<float>>("biases", attrs);
  }

  const Tensor *InputX() const { return input_x_; }

  const Tensor *InputBias() const { return input_bias_; }

  Tensor *Out() const { return out_; }

  const bool &Inplace() const { return inplace_; }

  const bool &HasBias() const { return has_bias_; }

  const vector<float> &Scales() const { return scales_; }

  const vector<float> &Biases() const { return biases_; }

 private:
  Tensor *input_x_;
  Tensor *input_bias_;
  Tensor *out_;
  bool inplace_;
  bool has_bias_;
  vector<float> scales_;
  vector<float> biases_;
};
#endif

#ifdef SLICE_OP
class SliceParam : public OpParam {
 public:
  SliceParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
             const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    input_shape_ = InputShapeFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
    axis_ = GetAttr<int>("axis", attrs);
    slice_points_ = GetAttr<vector<int>>("slice_points", attrs);
    inplace_ = GetAttr<bool>("inplace", attrs);
  }

  const Tensor *InputX() const { return input_x_; }

  const Tensor *InputShape() const { return input_shape_; }

  Tensor *Out() const { return out_; }

  const int &Axis() const { return axis_; }

  const vector<int> &SlicePoints() const { return slice_points_; }

  const bool &Inplace() const { return inplace_; }

 private:
  Tensor *input_x_;
  Tensor *input_shape_;
  Tensor *out_;
  int axis_;
  vector<int> slice_points_;
  bool inplace_;
};
#endif

#ifdef RESIZE_OP
class ResizeParam : public OpParam {
 public:
  ResizeParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
              const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    input_shape_ = InputShapeFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
    is_pyramid_test_ = GetAttr<bool>("is_pyramid_test", attrs);
    height_ = GetAttr<int>("height", attrs);
    width_ = GetAttr<int>("width", attrs);
    out_height_scale_ = GetAttr<float>("out_height_scale", attrs);
    out_width_scale_ = GetAttr<float>("out_width_scale", attrs);
  }

  const Tensor *InputX() const { return input_x_; }

  const Tensor *InputShape() const { return input_shape_; }

  Tensor *Out() const { return out_; }

  const bool &IsPyramidTest() const { return is_pyramid_test_; }

  const int &Height() const { return height_; }

  const int &Width() const { return width_; }

  const float &OutHeightScale() const { return out_height_scale_; }

  const float &OutWidthScale() const { return out_width_scale_; }

 private:
  Tensor *input_x_;
  Tensor *input_shape_;
  Tensor *out_;
  bool is_pyramid_test_;
  int height_;
  int width_;
  float out_height_scale_;
  float out_width_scale_;
};
#endif

#ifdef RELU_OP
/*
 * @b op 层实例化好这个 param 传递给 kernel 层使用
 * */
class ReluParam : public OpParam {
 public:
  ReluParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
            const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
  }

  const Tensor *InputX() const { return input_x_; }

  Tensor *Out() const { return out_; }

 private:
  Tensor *input_x_;
  Tensor *out_;
};
#endif

#ifdef PRELU_OP
class PReluParam : public OpParam {
 public:
  PReluParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
             const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
    slopes_ = GetAttr<vector<float>>("slopes", attrs);
  }

  const Tensor *InputX() const { return input_x_; }
  Tensor *Out() const { return out_; }
  const vector<float> &Slopes() const { return slopes_; }

 private:
  Tensor *input_x_;
  Tensor *out_;
  vector<float> slopes_;
};
#endif

class FusionFcParam : public OpParam {
 public:
  FusionFcParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
                const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    input_y_ = InputYFrom<LoDTensor>(inputs, scope);
    input_z_ = InputZFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
    x_num_col_dims_ = GetAttr<int>("x_num_col_dims", attrs);
    y_num_col_dims_ = GetAttr<int>("y_num_col_dims", attrs);
    axis_ = GetAttr<int>("axis", attrs);
  }
  const Tensor *InputX() const { return input_x_; }

#ifdef PADDLE_MOBILE_FPGA
  Tensor *InputY() const { return input_y_; }
#else
  const Tensor *InputY() const { return input_y_; }
#endif

  const Tensor *InputZ() const { return input_z_; }

  Tensor *Out() const { return out_; }

  const int &XNumColDims() const { return x_num_col_dims_; }

  const int &YNumColDims() const { return y_num_col_dims_; }

  const int &Axis() const { return axis_; }

 private:
  Tensor *input_x_;
  Tensor *input_y_;
  Tensor *input_z_;
  Tensor *out_;
  int x_num_col_dims_;
  int y_num_col_dims_;
  int axis_;
#ifdef PADDLE_MOBILE_FPGA

 private:
  fpga::ConvArgs fpga_conv_args;

 public:
  const fpga::ConvArgs &FpgaArgs() const { return fpga_conv_args; }
  void SetFpgaArgs(const fpga::ConvArgs &args) { fpga_conv_args = args; }
#endif
};

#ifdef FUSION_FCRELU_OP
using FusionFcReluParam = FusionFcParam;
#endif

class FusionConvAddParam : public OpParam {
 public:
  FusionConvAddParam(const VariableNameMap &inputs,
                     const VariableNameMap &outputs, const AttributeMap &attrs,
                     const Scope &scope) {
    bias_ = InputYFrom<LoDTensor>(inputs, scope);
    axis_ = GetAttr<int>("axis", attrs);
    filter_ = FilterFrom<LoDTensor>(inputs, scope);
    input_ = InputFrom<LoDTensor>(inputs, scope);
    output_ = OutFrom<LoDTensor>(outputs, scope);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
  }
  Tensor *Bias() const { return bias_; }

  const int &Axis() const { return axis_; }

  const Tensor *Input() const { return input_; }

#ifdef PADDLE_MOBILE_FPGA
  Tensor *Filter() const { return filter_; }
#else
  const Tensor *Filter() const { return filter_; }
#endif

  Tensor *Output() const { return output_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

 protected:
  Tensor *bias_;
  int axis_;
  Tensor *input_;
  Tensor *output_;
  Tensor *filter_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;
  int groups;
#ifdef PADDLE_MOBILE_FPGA

 private:
  fpga::ConvArgs fpga_conv_args;

 public:
  const fpga::ConvArgs &FpgaArgs() const { return fpga_conv_args; }
  void SetFpgaArgs(const fpga::ConvArgs &args) { fpga_conv_args = args; }
#endif
};

Print &operator<<(Print &printer, const FusionConvAddParam &conv_param);

#ifdef FUSION_CONVADDRELU_OP
class FusionConvAddReluParam : public FusionConvAddParam {
 public:
  FusionConvAddReluParam(const VariableNameMap &inputs,
                         const VariableNameMap &outputs,
                         const AttributeMap &attrs, const Scope &scope)
      : FusionConvAddParam(inputs, outputs, attrs, scope) {}
};
#endif

#ifdef FUSION_CONVADDBNRELU_OP
class FusionConvAddBNReluParam : public OpParam {
 public:
  FusionConvAddBNReluParam(const VariableNameMap &inputs,
                           const VariableNameMap &outputs,
                           const AttributeMap &attrs, const Scope &scope) {
    bias_ = InputYFrom<LoDTensor>(inputs, scope);
    axis_ = GetAttr<int>("axis", attrs);
    filter_ = FilterFrom<LoDTensor>(inputs, scope);
    input_ = InputFrom<LoDTensor>(inputs, scope);
    output_ = OutFrom<LoDTensor>(outputs, scope);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
    input_bias_ = InputBiasFrom<LoDTensor>(inputs, scope);
    input_mean_ = InputMeanFrom<LoDTensor>(inputs, scope);
    input_scale_ = InputScaleFrom<LoDTensor>(inputs, scope);
    input_variance_ = InputVarianceFrom<LoDTensor>(inputs, scope);
    epsilon_ = GetAttr<float>("epsilon", attrs);
    momentum_ = GetAttr<float>("momentum", attrs);
    //    is_test_ = GetAttr<bool>("is_test", attrs);
  }
  Tensor *Bias() const { return bias_; }

  const int &Axis() const { return axis_; }

  const Tensor *Input() const { return input_; }

#ifdef PADDLE_MOBILE_FPGA
  Tensor *Filter() const { return filter_; }
#else
  const Tensor *Filter() const { return filter_; }
#endif

  Tensor *Output() const { return output_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

  const Tensor *InputBias() const { return input_bias_; }

  const Tensor *InputMean() const { return input_mean_; }

  const Tensor *InputScale() const { return input_scale_; }

  const Tensor *InputVariance() const { return input_variance_; }

  const float &Epsilon() const { return epsilon_; }

  const float &Momentum() const { return momentum_; }

  const bool &IsTest() const { return is_test_; }

  void SetNewScale(Tensor *new_scale) { new_scale_ = new_scale; }

  void SetNewBias(Tensor *new_bias) { new_bias_ = new_bias; }

  const Tensor *NewScale() const { return new_scale_; }

  const Tensor *NewBias() const { return new_bias_; }

 protected:
  Tensor *bias_;
  int axis_;
  Tensor *input_;
  Tensor *output_;
  Tensor *filter_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;
  int groups;
  Tensor *input_bias_;
  Tensor *input_mean_;
  Tensor *input_scale_;
  Tensor *input_variance_;
  float epsilon_;
  float momentum_;
  bool is_test_;
  Tensor *new_bias_;
  Tensor *new_scale_;
#ifdef PADDLE_MOBILE_FPGA

 private:
  fpga::ConvArgs fpga_conv_args;

 public:
  const fpga::ConvArgs &FpgaArgs() const { return fpga_conv_args; }
  void SetFpgaArgs(const fpga::ConvArgs &args) { fpga_conv_args = args; }
#endif
};
#endif

#ifdef FUSION_CONVADDBN_OP
class FusionConvAddBNParam : public OpParam {
 public:
  FusionConvAddBNParam(const VariableNameMap &inputs,
                       const VariableNameMap &outputs,
                       const AttributeMap &attrs, const Scope &scope) {
    bias_ = InputYFrom<LoDTensor>(inputs, scope);
    axis_ = GetAttr<int>("axis", attrs);
    filter_ = FilterFrom<LoDTensor>(inputs, scope);
    input_ = InputFrom<LoDTensor>(inputs, scope);
    output_y_ = OutputYFrom<LoDTensor>(outputs, scope);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
    input_bias_ = InputBiasFrom<LoDTensor>(inputs, scope);
    input_mean_ = InputMeanFrom<LoDTensor>(inputs, scope);
    input_scale_ = InputScaleFrom<LoDTensor>(inputs, scope);
    input_variance_ = InputVarianceFrom<LoDTensor>(inputs, scope);
    epsilon_ = GetAttr<float>("epsilon", attrs);
    momentum_ = GetAttr<float>("momentum", attrs);
    //    is_test_ = GetAttr<bool>("is_test", attrs);
  }
  Tensor *Bias() const { return bias_; }

  const int &Axis() const { return axis_; }

  const Tensor *Input() const { return input_; }

#ifdef PADDLE_MOBILE_FPGA
  Tensor *Filter() const { return filter_; }
#else
  const Tensor *Filter() const { return filter_; }
#endif
  Tensor *Output() const { return output_y_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

  const Tensor *InputBias() const { return input_bias_; }

  const Tensor *InputMean() const { return input_mean_; }

  const Tensor *InputScale() const { return input_scale_; }

  const Tensor *InputVariance() const { return input_variance_; }

  const float &Epsilon() const { return epsilon_; }

  const float &Momentum() const { return momentum_; }

  const bool &IsTest() const { return is_test_; }

  void SetNewScale(Tensor *new_scale) { new_scale_ = new_scale; }

  void SetNewBias(Tensor *new_bias) { new_bias_ = new_bias; }

  const Tensor *NewScale() const { return new_scale_; }

  const Tensor *NewBias() const { return new_bias_; }

 protected:
  Tensor *bias_;
  int axis_;
  Tensor *input_;
  Tensor *output_y_;
  Tensor *filter_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;
  int groups;
  Tensor *input_bias_;
  Tensor *input_mean_;
  Tensor *input_scale_;
  Tensor *input_variance_;
  float epsilon_;
  float momentum_;
  bool is_test_;
  Tensor *new_bias_;
  Tensor *new_scale_;
#ifdef PADDLE_MOBILE_FPGA

 private:
  fpga::ConvArgs fpga_conv_args;

 public:
  const fpga::ConvArgs &FpgaArgs() const { return fpga_conv_args; }
  void SetFpgaArgs(const fpga::ConvArgs &args) { fpga_conv_args = args; }
#endif
};
#endif

#ifdef FUSION_DWCONVBNRELU_OP
class FusionDWConvBNReluParam : public OpParam {
 public:
  FusionDWConvBNReluParam(const VariableNameMap &inputs,
                          const VariableNameMap &outputs,
                          const AttributeMap &attrs, const Scope &scope) {
    filter_ = FilterFrom<LoDTensor>(inputs, scope);
    input_ = InputFrom<LoDTensor>(inputs, scope);
    output_ = OutFrom<LoDTensor>(outputs, scope);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
    input_bias_ = InputBiasFrom<LoDTensor>(inputs, scope);
    input_mean_ = InputMeanFrom<LoDTensor>(inputs, scope);
    input_scale_ = InputScaleFrom<LoDTensor>(inputs, scope);
    input_variance_ = InputVarianceFrom<LoDTensor>(inputs, scope);
    epsilon_ = GetAttr<float>("epsilon", attrs);
    momentum_ = GetAttr<float>("momentum", attrs);
    //    is_test_ = GetAttr<bool>("is_test", attrs);
  }

  const Tensor *Input() const { return input_; }

  const Tensor *Filter() const { return filter_; }

  Tensor *Output() const { return output_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

  const Tensor *InputBias() const { return input_bias_; }

  const Tensor *InputMean() const { return input_mean_; }

  const Tensor *InputScale() const { return input_scale_; }

  const Tensor *InputVariance() const { return input_variance_; }

  const float &Epsilon() const { return epsilon_; }

  const float &Momentum() const { return momentum_; }

  const bool &IsTest() const { return is_test_; }

  void SetNewScale(Tensor *new_scale) { new_scale_ = new_scale; }

  void SetNewBias(Tensor *new_bias) { new_bias_ = new_bias; }

  const Tensor *NewScale() const { return new_scale_; }

  const Tensor *NewBias() const { return new_bias_; }

 protected:
  Tensor *input_;
  Tensor *output_;
  Tensor *filter_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;
  int groups;
  Tensor *input_bias_;
  Tensor *input_mean_;
  Tensor *input_scale_;
  Tensor *input_variance_;
  float epsilon_;
  float momentum_;
  bool is_test_;
  Tensor *new_bias_;
  Tensor *new_scale_;
};

Print &operator<<(Print &printer, const FusionConvAddParam &conv_param);
#endif

#ifdef FUSION_CONVBNRELU_OP
class FusionConvBNReluParam : public OpParam {
 public:
  FusionConvBNReluParam(const VariableNameMap &inputs,
                        const VariableNameMap &outputs,
                        const AttributeMap &attrs, const Scope &scope) {
    filter_ = FilterFrom<LoDTensor>(inputs, scope);
    input_ = InputFrom<LoDTensor>(inputs, scope);
    output_ = OutFrom<LoDTensor>(outputs, scope);

    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
    input_bias_ = InputBiasFrom<LoDTensor>(inputs, scope);
    input_mean_ = InputMeanFrom<LoDTensor>(inputs, scope);
    input_scale_ = InputScaleFrom<LoDTensor>(inputs, scope);
    input_variance_ = InputVarianceFrom<LoDTensor>(inputs, scope);
    epsilon_ = GetAttr<float>("epsilon", attrs);
    momentum_ = GetAttr<float>("momentum", attrs);
    //    is_test_ = GetAttr<bool>("is_test", attrs);
  }

  const Tensor *Input() const { return input_; }

  const Tensor *Filter() const { return filter_; }

  Tensor *Output() const { return output_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

  const Tensor *InputBias() const { return input_bias_; }

  const Tensor *InputMean() const { return input_mean_; }

  const Tensor *InputScale() const { return input_scale_; }

  const Tensor *InputVariance() const { return input_variance_; }

  const float &Epsilon() const { return epsilon_; }

  const float &Momentum() const { return momentum_; }

  const bool &IsTest() const { return is_test_; }

  void SetNewScale(Tensor *new_scale) { new_scale_ = new_scale; }

  void SetNewBias(Tensor *new_bias) { new_bias_ = new_bias; }

  const Tensor *NewScale() const { return new_scale_; }

  const Tensor *NewBias() const { return new_bias_; }

 protected:
  Tensor *input_;
  Tensor *output_;
  Tensor *filter_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;
  int groups;
  Tensor *input_bias_;
  Tensor *input_mean_;
  Tensor *input_scale_;
  Tensor *input_variance_;
  float epsilon_;
  float momentum_;
  bool is_test_;
  Tensor *new_bias_;
  Tensor *new_scale_;
};
#endif

#ifdef IM2SEQUENCE_OP
class Im2SequenceParam : public OpParam {
 public:
  Im2SequenceParam(const VariableNameMap &inputs,
                   const VariableNameMap &outputs, const AttributeMap &attrs,
                   const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
    kernels_ = GetAttr<vector<int>>("kernels", attrs);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
  }

  const Tensor *Input() const { return input_x_; }

  Tensor *Output() const { return out_; }

  const vector<int> &Kernels() const { return kernels_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

 private:
  Tensor *input_x_;
  Tensor *out_;
  vector<int> kernels_;
  vector<int> strides_;
  vector<int> paddings_;
};
#endif

#ifdef DROPOUT_OP
class DropoutParam : public OpParam {
 public:
  DropoutParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
               const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<LoDTensor>(inputs, scope);
    out_ = OutFrom<LoDTensor>(outputs, scope);
  }

  const Tensor *InputX() const { return input_x_; }

  Tensor *Out() const { return out_; }

 private:
  Tensor *input_x_;
  Tensor *out_;
};
#endif

#ifdef CONV_TRANSPOSE
class ConvTransposeParam : public OpParam {
 public:
  ConvTransposeParam(const VariableNameMap &inputs,
                     const VariableNameMap &outputs, const AttributeMap &attrs,
                     const Scope &scope) {
    filter_ = FilterFrom<LoDTensor>(inputs, scope);
    input_ = InputFrom<LoDTensor>(inputs, scope);
    output_ = OutputFrom<LoDTensor>(outputs, scope);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
  }

  const Tensor *Input() const { return input_; }

  const Tensor *Filter() const { return filter_; }

  Tensor *Output() const { return output_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

 private:
  Tensor *input_;
  Tensor *output_;
  Tensor *filter_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;
  int groups;
};
#endif

}  // namespace operators
}  // namespace paddle_mobile
