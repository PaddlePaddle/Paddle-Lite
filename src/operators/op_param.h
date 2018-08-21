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
#include "common/types.h"
#include "framework/lod_tensor.h"
#include "framework/scope.h"
#include "framework/tensor.h"
#include "framework/variable.h"
#ifdef PADDLE_MOBILE_FPGA
#include "fpga/api.h"
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

template <typename Dtype>
struct DtypeTensorTrait {
  typedef void ptype;
  typedef void rtype;
};

template <>
struct DtypeTensorTrait<CPU> {
  // This is the type we obtained in variable.
  typedef framework::LoDTensor gtype;
  // This type will be the parent class type
  // or the same type.
  typedef framework::Tensor rtype;
};

template <>
struct DtypeTensorTrait<FPGA> {
  // This is the type we obtained in variable.
  typedef framework::LoDTensor gtype;
  // This type will be the parent class type
  // or the same type.
  typedef framework::Tensor rtype;
};

template <>
struct DtypeTensorTrait<GPU_MALI> {
  // This is the type we obtained in variable.
  typedef framework::LoDTensor gtype;
  // This type will be the parent class type
  // or the same type.
  typedef framework::Tensor rtype;
};

class OpParam {
 protected:
  template <typename T>
  static T *InputAlphaFrom(const VariableNameMap &inputs, const Scope &scope) {
    return GetVarValue<T>("Alpha", inputs, scope);
  }

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
template <typename Dtype>
class ConvParam : OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  ConvParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
            const AttributeMap &attrs, const Scope &scope) {
    filter_ = FilterFrom<GType>(inputs, scope);
    input_ = InputFrom<GType>(inputs, scope);
    output_ = OutputFrom<GType>(outputs, scope);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
  }

  const RType *Input() const { return input_; }

  RType *Filter() const { return filter_; }

  RType *Output() const { return output_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

 private:
  RType *input_;
  RType *output_;
  RType *filter_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;
  int groups;
};
template <typename Dtype>
Print &operator<<(Print &printer, const ConvParam<Dtype> &conv_param);
#endif

template <typename Dtype>
class ElementwiseAddParam : OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  ElementwiseAddParam(const VariableNameMap &inputs,
                      const VariableNameMap &outputs, const AttributeMap &attrs,
                      const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    input_y_ = InputYFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
    axis_ = GetAttr<int>("axis", attrs);
  }

  const RType *InputX() const { return input_x_; }

  const RType *InputY() const { return input_y_; }

  RType *Out() const { return out_; }

  const int &Axis() const { return axis_; }

 private:
  RType *input_x_;
  RType *input_y_;
  RType *out_;
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
template <typename Dtype>
using ElementwiseAddReluParam = ElementwiseAddParam<Dtype>;
#endif

#ifdef MUL_OP
template <typename Dtype>
class MulParam : OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  MulParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
           const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    input_y_ = InputYFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
    x_num_col_dims_ = GetAttr<int>("x_num_col_dims", attrs);
    y_num_col_dims_ = GetAttr<int>("y_num_col_dims", attrs);
  }

  const RType *InputX() const { return input_x_; }

  const RType *InputY() const { return input_y_; }

  RType *Out() const { return out_; }

  const int &XNumColDims() const { return x_num_col_dims_; }

  const int &YNumColDims() const { return y_num_col_dims_; }

 private:
  RType *input_x_;
  RType *input_y_;
  RType *out_;
  int x_num_col_dims_;
  int y_num_col_dims_;
};
#endif

#ifdef CONCAT_OP
template <typename Dtype>
class ConcatParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  ConcatParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
              const AttributeMap &attrs, const Scope &scope) {
    inputs_ = InputMultiFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
    axis_ = GetAttr<int>("axis", attrs);
  }

  vector<GType *> Inputs() const { return inputs_; }

  RType *Out() const { return out_; }

  const int &Axis() const { return axis_; }

 private:
  vector<GType *> inputs_;
  RType *out_;
  int axis_;
};
#endif

#ifdef LRN_OP
template <typename Dtype>
class LrnParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  LrnParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
           const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
    mid_out_ = MidOutFrom<GType>(outputs, scope);
    n_ = GetAttr<int>("n", attrs);
    alpha_ = GetAttr<float>("alpha", attrs);
    beta_ = GetAttr<float>("beta", attrs);
    k_ = GetAttr<float>("k", attrs);
    data_format_ = GetAttr<string>("data_format", attrs);
  }

  const RType *InputX() const { return input_x_; }

  RType *Out() const { return out_; }

  RType *MidOut() const { return mid_out_; }

  const int &N() const { return n_; }

  const float &Alpha() const { return alpha_; }

  const float &Beta() const { return beta_; }

  const float &K() const { return k_; }

  const string &DataFormat() const { return data_format_; }

 private:
  RType *input_x_;
  RType *out_;
  RType *mid_out_;
  int n_;
  float alpha_;
  float beta_;
  float k_;
  string data_format_;
};
#endif

#ifdef BATCHNORM_OP
template <typename Dtype>
class BatchNormParam : OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  BatchNormParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
                 const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    output_y_ = OutputYFrom<GType>(outputs, scope);
    input_bias_ = InputBiasFrom<GType>(inputs, scope);
    input_mean_ = InputMeanFrom<GType>(inputs, scope);
    input_scale_ = InputScaleFrom<GType>(inputs, scope);
    input_variance_ = InputVarianceFrom<GType>(inputs, scope);
    epsilon_ = GetAttr<float>("epsilon", attrs);
    momentum_ = GetAttr<float>("momentum", attrs);
    //    is_test_ = GetAttr<bool>("is_test", attrs);
  }

  const RType *InputX() const { return input_x_; }

  RType *OutputY() const { return output_y_; }

  const RType *InputBias() const { return input_bias_; }

  const RType *InputMean() const { return input_mean_; }

  const RType *InputScale() const { return input_scale_; }

  const RType *InputVariance() const { return input_variance_; }

  const float &Epsilon() const { return epsilon_; }

  const float &Momentum() const { return momentum_; }

  const bool &IsTest() const { return is_test_; }

  const string &DataFormat() const { return data_format_; }

 private:
  RType *input_x_;
  RType *output_y_;
  RType *input_bias_;
  RType *input_mean_;
  RType *input_scale_;
  RType *input_variance_;
  float epsilon_;
  float momentum_;
  bool is_test_;
  string data_format_;
};
#endif

#ifdef POOL_OP
template <typename Dtype>
class PoolParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  PoolParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
            const AttributeMap &attrs, const Scope &scope) {
    input_ = InputXFrom<GType>(inputs, scope);

    output_ = OutFrom<GType>(outputs, scope);
    pooling_type_ = GetAttr<string>("pooling_type", attrs);
    ksize_ = GetAttr<vector<int>>("ksize", attrs);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    ceil_mode_ = GetAttr<bool>("ceil_mode", attrs);
    global_pooling_ = GetAttr<bool>("global_pooling", attrs);
  }

  const RType *Input() const { return input_; }

  RType *Output() const { return output_; }

  const string &PoolingType() const { return pooling_type_; }

  const vector<int> &Ksize() const { return ksize_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  bool isCeilMode() const { return ceil_mode_; }

  bool isGlobalPooling() const { return global_pooling_; }

 private:
  RType *input_;
  RType *output_;
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
template <typename Dtype>
class PriorBoxParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  PriorBoxParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
                const AttributeMap &attrs, const Scope &scope) {
    input_ = InputFrom<GType>(inputs, scope);
    input_image_ = InputImageFrom<GType>(inputs, scope);
    output_boxes_ = OutputBoxesFrom<GType>(outputs, scope);
    output_variances_ = OutputVariancesFrom<GType>(outputs, scope);
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
  const RType *Input() const { return input_; }

  const RType *InputImage() const { return input_image_; }

  RType *OutputBoxes() const { return output_boxes_; }

  RType *OutputVariances() const { return output_variances_; }

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
  RType *input_;
  RType *input_image_;
  RType *output_boxes_;
  RType *output_variances_;
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
template <typename Dtype>
class BoxCoderParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  BoxCoderParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
                const AttributeMap &attrs, const Scope &scope) {
    input_priorbox_ = InputPriorBoxFrom<GType>(inputs, scope);
    input_priorboxvar_ = InputPriorBoxVarFrom<GType>(inputs, scope);
    input_targetbox_ = InputTargetBoxFrom<GType>(inputs, scope);
    output_box_ = OutputBoxFrom<GType>(outputs, scope);
    code_type_ = GetAttr<std::string>("code_type", attrs);
  }
  const RType *InputPriorBox() const { return input_priorbox_; }

  const RType *InputPriorBoxVar() const { return input_priorboxvar_; }

  const RType *InputTargetBox() const { return input_targetbox_; }

  RType *OutputBox() const { return output_box_; }

  const std::string &CodeType() const { return code_type_; }

 private:
  RType *input_priorbox_;
  RType *input_priorboxvar_;
  RType *input_targetbox_;
  RType *output_box_;
  std::string code_type_;
};
#endif

#ifdef SOFTMAX_OP
template <typename Dtype>
class SoftmaxParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  SoftmaxParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
               const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
  }
  const RType *InputX() const { return input_x_; }
  RType *Out() const { return out_; }

 private:
  RType *input_x_;
  RType *out_;

#ifdef PADDLE_MOBILE_FPGA

 private:
  std::shared_ptr<RType> float_input_x_;
  fpga::BypassArgs fpga_bypass_args;

 public:
  RType *FloatInput() {
    return float_input_x_ == nullptr ? input_x_ : float_input_x_.get();
  }
  void SetFloatInput(Tensor *input) { float_input_x_.reset(input); }
  const fpga::BypassArgs &FpgaArgs() const { return fpga_bypass_args; }
  void SetFpgaArgs(const fpga::BypassArgs &args) { fpga_bypass_args = args; }
#endif
};
#endif

#ifdef SIGMOID_OP
template <typename Dtype>
class SigmoidParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  SigmoidParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
               const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
  }
  const RType *InputX() const { return input_x_; }
  RType *Out() const { return out_; }

 private:
  RType *input_x_;
  RType *out_;
};
#endif

#ifdef MULTICLASSNMS_OP
template <typename Dtype>
class MultiClassNMSParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  MultiClassNMSParam(const VariableNameMap &inputs,
                     const VariableNameMap &outputs, const AttributeMap &attrs,
                     const Scope &scope) {
    input_bboxes_ = InputBBoxesFrom<GType>(inputs, scope);
    input_scores_ = InputScoresFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
    background_label_ = GetAttr<int>("background_label", attrs);
    nms_top_k_ = GetAttr<int>("nms_top_k", attrs);
    keep_top_k_ = GetAttr<int>("keep_top_k", attrs);
    nms_threshold_ = GetAttr<float>("nms_threshold", attrs);
    nms_eta_ = GetAttr<float>("nms_eta", attrs);
    score_threshold_ = GetAttr<float>("score_threshold", attrs);
  }

  const RType *InputBBoxes() const { return input_bboxes_; }

  const RType *InputScores() const { return input_scores_; }

  RType *Out() const { return out_; }

  const int &BackGroundLabel() const { return background_label_; }

  const int &NMSTopK() const { return nms_top_k_; }

  const int &KeepTopK() const { return keep_top_k_; }

  const float &NMSThreshold() const { return nms_threshold_; }

  const float &NMSEta() const { return nms_eta_; }

  const float &ScoreThreshold() const { return score_threshold_; }

 private:
  RType *input_bboxes_;
  RType *input_scores_;
  RType *out_;
  int background_label_;
  int nms_top_k_;
  int keep_top_k_;
  float nms_threshold_;
  float nms_eta_;
  float score_threshold_;
};
#endif

template <typename Dtype>
class FeedParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  FeedParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
            const AttributeMap &attrs, Scope *scope) {
    input_x_ = InputXFrom<GType>(inputs, *scope);
    out_ = OutFrom<GType>(outputs, *scope);
    auto var = scope->Var("batch_size");
    batch_size = var->GetValue<int>();
  }
  const RType *InputX() const { return input_x_; }
  RType *Out() const { return out_; }
  const int BatchSize() const { return batch_size; }

 private:
  RType *input_x_;
  RType *out_;
  int batch_size;
};

template <typename Dtype>
class FetchParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  FetchParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
             const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
  }
  const RType *InputX() const { return input_x_; }
  RType *Out() const { return out_; }

 private:
  RType *input_x_;
  RType *out_;
};

#ifdef TRANSPOSE_OP
template <typename Dtype>
class TransposeParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  TransposeParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
                 const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
    axis_ = GetAttr<vector<int>>("axis", attrs);
  }

  const RType *InputX() const { return input_x_; }

  RType *Out() const { return out_; }

  const vector<int> &Axis() const { return axis_; }

 private:
  RType *input_x_;
  RType *out_;
  vector<int> axis_;
};
#endif

#ifdef RESHAPE_OP
template <typename Dtype>
class ReshapeParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  ReshapeParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
               const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    input_shape_ = InputShapeFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
    shape_ = GetAttr<vector<int>>("shape", attrs);
    inplace_ = GetAttr<bool>("inplace", attrs);
  }

  const RType *InputX() const { return input_x_; }

  const RType *InputShape() const { return input_shape_; }

  RType *Out() const { return out_; }

  const vector<int> &Shape() const { return shape_; }

  const bool &Inplace() const { return inplace_; }

 private:
  RType *input_x_;
  RType *input_shape_;
  RType *out_;
  vector<int> shape_;
  bool inplace_;
};
#endif

#ifdef SCALE_OP
template <typename Dtype>
class ScaleParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  ScaleParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
             const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    input_bias_ = InputBiasFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
    inplace_ = GetAttr<bool>("inplace", attrs);
    has_bias_ = GetAttr<bool>("has_bias", attrs);
    scales_ = GetAttr<vector<float>>("scales", attrs);
    biases_ = GetAttr<vector<float>>("biases", attrs);
  }

  const RType *InputX() const { return input_x_; }

  const RType *InputBias() const { return input_bias_; }

  RType *Out() const { return out_; }

  const bool &Inplace() const { return inplace_; }

  const bool &HasBias() const { return has_bias_; }

  const vector<float> &Scales() const { return scales_; }

  const vector<float> &Biases() const { return biases_; }

 private:
  RType *input_x_;
  RType *input_bias_;
  RType *out_;
  bool inplace_;
  bool has_bias_;
  vector<float> scales_;
  vector<float> biases_;
};
#endif

#ifdef SLICE_OP
template <typename Dtype>
class SliceParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  SliceParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
             const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    input_shape_ = InputShapeFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
    axis_ = GetAttr<int>("axis", attrs);
    slice_points_ = GetAttr<vector<int>>("slice_points", attrs);
    inplace_ = GetAttr<bool>("inplace", attrs);
  }

  const RType *InputX() const { return input_x_; }

  const RType *InputShape() const { return input_shape_; }

  RType *Out() const { return out_; }

  const int &Axis() const { return axis_; }

  const vector<int> &SlicePoints() const { return slice_points_; }

  const bool &Inplace() const { return inplace_; }

 private:
  RType *input_x_;
  RType *input_shape_;
  RType *out_;
  int axis_;
  vector<int> slice_points_;
  bool inplace_;
};
#endif

#ifdef RESIZE_OP
template <typename Dtype>
class ResizeParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  ResizeParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
              const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    input_shape_ = InputShapeFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
    is_pyramid_test_ = GetAttr<bool>("is_pyramid_test", attrs);
    height_ = GetAttr<int>("height", attrs);
    width_ = GetAttr<int>("width", attrs);
    out_height_scale_ = GetAttr<float>("out_height_scale", attrs);
    out_width_scale_ = GetAttr<float>("out_width_scale", attrs);
  }

  const RType *InputX() const { return input_x_; }

  const RType *InputShape() const { return input_shape_; }

  RType *Out() const { return out_; }

  const bool &IsPyramidTest() const { return is_pyramid_test_; }

  const int &Height() const { return height_; }

  const int &Width() const { return width_; }

  const float &OutHeightScale() const { return out_height_scale_; }

  const float &OutWidthScale() const { return out_width_scale_; }

 private:
  RType *input_x_;
  RType *input_shape_;
  RType *out_;
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
template <typename Dtype>
class ReluParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  ReluParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
            const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
  }

  const RType *InputX() const { return input_x_; }

  RType *Out() const { return out_; }

 private:
  RType *input_x_;
  RType *out_;
};
#endif

#ifdef PRELU_OP
template <typename Dtype>
class PReluParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  PReluParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
             const AttributeMap &attrs, const Scope &scope) {
    DLOG << "PReluParam inputs before";
    input_x_ = InputXFrom<GType>(inputs, scope);
    alpha_ = InputAlphaFrom<GType>(inputs, scope);
    framework::DDim dims = alpha_->dims();
    out_ = OutFrom<GType>(outputs, scope);
    mode_ = GetAttr<std::string>("mode", attrs);
    DLOG << "PReluParam mode after" << mode_;
  }
  const RType *InputX() const { return input_x_; }
  const RType *InputAlpha() const { return alpha_; }
  RType *Out() const { return out_; }
  const std::string &Mode() const { return mode_; }

 private:
  RType *input_x_;
  RType *out_;
  RType *alpha_;
  std::string mode_;
};
#endif

template <typename Dtype>
class FusionFcParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  FusionFcParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
                const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    input_y_ = InputYFrom<GType>(inputs, scope);
    input_z_ = InputZFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
    x_num_col_dims_ = GetAttr<int>("x_num_col_dims", attrs);
    y_num_col_dims_ = GetAttr<int>("y_num_col_dims", attrs);
    axis_ = GetAttr<int>("axis", attrs);
  }
  const RType *InputX() const { return input_x_; }

#ifdef PADDLE_MOBILE_FPGA
  RType *InputY() const { return input_y_; }
#else
  const RType *InputY() const { return input_y_; }
#endif

  const RType *InputZ() const { return input_z_; }

  RType *Out() const { return out_; }

  const int &XNumColDims() const { return x_num_col_dims_; }

  const int &YNumColDims() const { return y_num_col_dims_; }

  const int &Axis() const { return axis_; }

 private:
  RType *input_x_;
  RType *input_y_;
  RType *input_z_;
  RType *out_;
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
template <typename DeviceType>
using FusionFcReluParam = FusionFcParam<DeviceType>;
#endif

template <typename Dtype>
class FusionConvAddParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  FusionConvAddParam(const VariableNameMap &inputs,
                     const VariableNameMap &outputs, const AttributeMap &attrs,
                     const Scope &scope) {
    bias_ = InputYFrom<GType>(inputs, scope);
    axis_ = GetAttr<int>("axis", attrs);
    filter_ = FilterFrom<GType>(inputs, scope);
    input_ = InputFrom<GType>(inputs, scope);
    output_ = OutFrom<GType>(outputs, scope);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
  }
  RType *Bias() const { return bias_; }

  const int &Axis() const { return axis_; }

  const RType *Input() const { return input_; }

#ifdef PADDLE_MOBILE_FPGA
  RType *Filter() const { return filter_; }
#else
  const RType *Filter() const { return filter_; }
#endif

  RType *Output() const { return output_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

 protected:
  RType *bias_;
  int axis_;
  RType *input_;
  RType *output_;
  RType *filter_;
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

template <typename Dtype>
Print &operator<<(Print &printer, const FusionConvAddParam<Dtype> &conv_param);

#ifdef FUSION_CONVADDRELU_OP
template <typename DeviceType>
class FusionConvAddReluParam : public FusionConvAddParam<DeviceType> {
 public:
  FusionConvAddReluParam(const VariableNameMap &inputs,
                         const VariableNameMap &outputs,
                         const AttributeMap &attrs, const Scope &scope)
      : FusionConvAddParam<DeviceType>(inputs, outputs, attrs, scope) {}
};
#endif

#ifdef FUSION_CONVADDBNRELU_OP
template <typename Dtype>
class FusionConvAddBNReluParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  FusionConvAddBNReluParam(const VariableNameMap &inputs,
                           const VariableNameMap &outputs,
                           const AttributeMap &attrs, const Scope &scope) {
    bias_ = InputYFrom<GType>(inputs, scope);
    axis_ = GetAttr<int>("axis", attrs);
    filter_ = FilterFrom<GType>(inputs, scope);
    input_ = InputFrom<GType>(inputs, scope);
    output_ = OutFrom<GType>(outputs, scope);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
    input_bias_ = InputBiasFrom<GType>(inputs, scope);
    input_mean_ = InputMeanFrom<GType>(inputs, scope);
    input_scale_ = InputScaleFrom<GType>(inputs, scope);
    input_variance_ = InputVarianceFrom<GType>(inputs, scope);
    epsilon_ = GetAttr<float>("epsilon", attrs);
    momentum_ = GetAttr<float>("momentum", attrs);
    //    is_test_ = GetAttr<bool>("is_test", attrs);
  }
  RType *Bias() const { return bias_; }

  const int &Axis() const { return axis_; }

  const RType *Input() const { return input_; }

#ifdef PADDLE_MOBILE_FPGA
  RType *Filter() const { return filter_; }
#else
  const RType *Filter() const { return filter_; }
#endif

  RType *Output() const { return output_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

  const RType *InputBias() const { return input_bias_; }

  const RType *InputMean() const { return input_mean_; }

  const RType *InputScale() const { return input_scale_; }

  const RType *InputVariance() const { return input_variance_; }

  const float &Epsilon() const { return epsilon_; }

  const float &Momentum() const { return momentum_; }

  const bool &IsTest() const { return is_test_; }

  void SetNewScale(RType *new_scale) { new_scale_ = new_scale; }

  void SetNewBias(RType *new_bias) { new_bias_ = new_bias; }

  const RType *NewScale() const { return new_scale_; }

  const RType *NewBias() const { return new_bias_; }

 protected:
  RType *bias_;
  int axis_;
  RType *input_;
  RType *output_;
  RType *filter_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;
  int groups;
  RType *input_bias_;
  RType *input_mean_;
  RType *input_scale_;
  RType *input_variance_;
  float epsilon_;
  float momentum_;
  bool is_test_;
  RType *new_bias_;
  RType *new_scale_;
#ifdef PADDLE_MOBILE_FPGA

 private:
  fpga::ConvArgs fpga_conv_args;

 public:
  const fpga::ConvArgs &FpgaArgs() const { return fpga_conv_args; }
  void SetFpgaArgs(const fpga::ConvArgs &args) { fpga_conv_args = args; }
#endif
};
#endif

#ifdef FUSION_CONVBN_OP
template <typename Dtype>
class FusionConvBNParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  FusionConvBNParam(const VariableNameMap &inputs,
                    const VariableNameMap &outputs, const AttributeMap &attrs,
                    const Scope &scope) {
    filter_ = FilterFrom<GType>(inputs, scope);
    input_ = InputFrom<GType>(inputs, scope);
    output_y_ = OutputYFrom<GType>(outputs, scope);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
    input_bias_ = InputBiasFrom<GType>(inputs, scope);
    input_mean_ = InputMeanFrom<GType>(inputs, scope);
    input_scale_ = InputScaleFrom<GType>(inputs, scope);
    input_variance_ = InputVarianceFrom<GType>(inputs, scope);
    epsilon_ = GetAttr<float>("epsilon", attrs);
    momentum_ = GetAttr<float>("momentum", attrs);
    //    is_test_ = GetAttr<bool>("is_test", attrs);
  }

  const RType *Input() const { return input_; }

#ifdef PADDLE_MOBILE_FPGA
  RType *Filter() const { return filter_; }
#else
  const RType *Filter() const { return filter_; }
#endif
  RType *Output() const { return output_y_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

  const RType *InputBias() const { return input_bias_; }

  const RType *InputMean() const { return input_mean_; }

  const RType *InputScale() const { return input_scale_; }

  const RType *InputVariance() const { return input_variance_; }

  const float &Epsilon() const { return epsilon_; }

  const float &Momentum() const { return momentum_; }

  const bool &IsTest() const { return is_test_; }

  void SetNewScale(RType *new_scale) { new_scale_ = new_scale; }

  void SetNewBias(RType *new_bias) { new_bias_ = new_bias; }

  const RType *NewScale() const { return new_scale_; }

  const RType *NewBias() const { return new_bias_; }

 protected:
  RType *input_;
  RType *output_y_;
  RType *filter_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;
  int groups;
  RType *input_bias_;
  RType *input_mean_;
  RType *input_scale_;
  RType *input_variance_;
  float epsilon_;
  float momentum_;
  bool is_test_;
  RType *new_bias_;
  RType *new_scale_;
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
template <typename Dtype>
class FusionConvAddBNParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  FusionConvAddBNParam(const VariableNameMap &inputs,
                       const VariableNameMap &outputs,
                       const AttributeMap &attrs, const Scope &scope) {
    bias_ = InputYFrom<GType>(inputs, scope);
    axis_ = GetAttr<int>("axis", attrs);
    filter_ = FilterFrom<GType>(inputs, scope);
    input_ = InputFrom<GType>(inputs, scope);
    output_y_ = OutputYFrom<GType>(outputs, scope);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
    input_bias_ = InputBiasFrom<GType>(inputs, scope);
    input_mean_ = InputMeanFrom<GType>(inputs, scope);
    input_scale_ = InputScaleFrom<GType>(inputs, scope);
    input_variance_ = InputVarianceFrom<GType>(inputs, scope);
    epsilon_ = GetAttr<float>("epsilon", attrs);
    momentum_ = GetAttr<float>("momentum", attrs);
    //    is_test_ = GetAttr<bool>("is_test", attrs);
  }
  RType *Bias() const { return bias_; }

  const int &Axis() const { return axis_; }

  const RType *Input() const { return input_; }

#ifdef PADDLE_MOBILE_FPGA
  RType *Filter() const { return filter_; }
#else
  const RType *Filter() const { return filter_; }
#endif
  RType *Output() const { return output_y_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

  const RType *InputBias() const { return input_bias_; }

  const RType *InputMean() const { return input_mean_; }

  const RType *InputScale() const { return input_scale_; }

  const RType *InputVariance() const { return input_variance_; }

  const float &Epsilon() const { return epsilon_; }

  const float &Momentum() const { return momentum_; }

  const bool &IsTest() const { return is_test_; }

  void SetNewScale(RType *new_scale) { new_scale_ = new_scale; }

  void SetNewBias(RType *new_bias) { new_bias_ = new_bias; }

  const RType *NewScale() const { return new_scale_; }

  const RType *NewBias() const { return new_bias_; }

 protected:
  RType *bias_;
  int axis_;
  RType *input_;
  RType *output_y_;
  RType *filter_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;
  int groups;
  RType *input_bias_;
  RType *input_mean_;
  RType *input_scale_;
  RType *input_variance_;
  float epsilon_;
  float momentum_;
  bool is_test_;
  RType *new_bias_;
  RType *new_scale_;
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
template <typename Dtype>
class FusionDWConvBNReluParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  FusionDWConvBNReluParam(const VariableNameMap &inputs,
                          const VariableNameMap &outputs,
                          const AttributeMap &attrs, const Scope &scope) {
    filter_ = FilterFrom<GType>(inputs, scope);
    input_ = InputFrom<GType>(inputs, scope);
    output_ = OutFrom<GType>(outputs, scope);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
    input_bias_ = InputBiasFrom<GType>(inputs, scope);
    input_mean_ = InputMeanFrom<GType>(inputs, scope);
    input_scale_ = InputScaleFrom<GType>(inputs, scope);
    input_variance_ = InputVarianceFrom<GType>(inputs, scope);
    epsilon_ = GetAttr<float>("epsilon", attrs);
    momentum_ = GetAttr<float>("momentum", attrs);
    //    is_test_ = GetAttr<bool>("is_test", attrs);
  }

  const RType *Input() const { return input_; }

  const RType *Filter() const { return filter_; }

  RType *Output() const { return output_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

  const RType *InputBias() const { return input_bias_; }

  const RType *InputMean() const { return input_mean_; }

  const RType *InputScale() const { return input_scale_; }

  const RType *InputVariance() const { return input_variance_; }

  const float &Epsilon() const { return epsilon_; }

  const float &Momentum() const { return momentum_; }

  const bool &IsTest() const { return is_test_; }

  void SetNewScale(RType *new_scale) { new_scale_ = new_scale; }

  void SetNewBias(RType *new_bias) { new_bias_ = new_bias; }

  const RType *NewScale() const { return new_scale_; }

  const RType *NewBias() const { return new_bias_; }

 protected:
  RType *input_;
  RType *output_;
  RType *filter_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;
  int groups;
  RType *input_bias_;
  RType *input_mean_;
  RType *input_scale_;
  RType *input_variance_;
  float epsilon_;
  float momentum_;
  bool is_test_;
  RType *new_bias_;
  RType *new_scale_;
};

#endif

#ifdef FUSION_CONVBNRELU_OP
template <typename Dtype>
class FusionConvBNReluParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  FusionConvBNReluParam(const VariableNameMap &inputs,
                        const VariableNameMap &outputs,
                        const AttributeMap &attrs, const Scope &scope) {
    filter_ = FilterFrom<GType>(inputs, scope);
    input_ = InputFrom<GType>(inputs, scope);
    output_ = OutFrom<GType>(outputs, scope);

    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
    input_bias_ = InputBiasFrom<GType>(inputs, scope);
    input_mean_ = InputMeanFrom<GType>(inputs, scope);
    input_scale_ = InputScaleFrom<GType>(inputs, scope);
    input_variance_ = InputVarianceFrom<GType>(inputs, scope);
    epsilon_ = GetAttr<float>("epsilon", attrs);
    momentum_ = GetAttr<float>("momentum", attrs);
    //    is_test_ = GetAttr<bool>("is_test", attrs);
  }

  const RType *Input() const { return input_; }

#ifdef PADDLE_MOBILE_FPGA
  RType *Filter() const { return filter_; }
#else
  const RType *Filter() const { return filter_; }
#endif

  RType *Output() const { return output_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

  const RType *InputBias() const { return input_bias_; }

  const RType *InputMean() const { return input_mean_; }

  const RType *InputScale() const { return input_scale_; }

  const RType *InputVariance() const { return input_variance_; }

  const float &Epsilon() const { return epsilon_; }

  const float &Momentum() const { return momentum_; }

  const bool &IsTest() const { return is_test_; }

  void SetNewScale(RType *new_scale) { new_scale_ = new_scale; }

  void SetNewBias(RType *new_bias) { new_bias_ = new_bias; }

  const RType *NewScale() const { return new_scale_; }

  const RType *NewBias() const { return new_bias_; }

 protected:
  RType *input_;
  RType *output_;
  RType *filter_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;
  int groups;
  RType *input_bias_;
  RType *input_mean_;
  RType *input_scale_;
  RType *input_variance_;
  float epsilon_;
  float momentum_;
  bool is_test_;
  RType *new_bias_;
  RType *new_scale_;
#ifdef PADDLE_MOBILE_FPGA

 private:
  fpga::ConvArgs fpga_conv_args;

 public:
  const fpga::ConvArgs &FpgaArgs() const { return fpga_conv_args; }
  void SetFpgaArgs(const fpga::ConvArgs &args) { fpga_conv_args = args; }
#endif
};
#endif

#ifdef IM2SEQUENCE_OP
template <typename Dtype>
class Im2SequenceParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  Im2SequenceParam(const VariableNameMap &inputs,
                   const VariableNameMap &outputs, const AttributeMap &attrs,
                   const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
    kernels_ = GetAttr<vector<int>>("kernels", attrs);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
  }

  const RType *Input() const { return input_x_; }

  RType *Output() const { return out_; }

  const vector<int> &Kernels() const { return kernels_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

 private:
  RType *input_x_;
  RType *out_;
  vector<int> kernels_;
  vector<int> strides_;
  vector<int> paddings_;
};
#endif

#ifdef DROPOUT_OP
template <typename Dtype>
class DropoutParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  DropoutParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
               const AttributeMap &attrs, const Scope &scope) {
    input_x_ = InputXFrom<GType>(inputs, scope);
    out_ = OutFrom<GType>(outputs, scope);
  }

  const RType *InputX() const { return input_x_; }

  RType *Out() const { return out_; }

 private:
  RType *input_x_;
  RType *out_;
};
#endif

#ifdef CONV_TRANSPOSE
template <typename Dtype>
class ConvTransposeParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;
  typedef typename DtypeTensorTrait<Dtype>::rtype RType;

 public:
  ConvTransposeParam(const VariableNameMap &inputs,
                     const VariableNameMap &outputs, const AttributeMap &attrs,
                     const Scope &scope) {
    filter_ = FilterFrom<GType>(inputs, scope);
    input_ = InputFrom<GType>(inputs, scope);
    output_ = OutputFrom<GType>(outputs, scope);
    strides_ = GetAttr<vector<int>>("strides", attrs);
    paddings_ = GetAttr<vector<int>>("paddings", attrs);
    dilations_ = GetAttr<vector<int>>("dilations", attrs);
    groups = GetAttr<int>("groups", attrs);
  }

  const RType *Input() const { return input_; }

  const RType *Filter() const { return filter_; }

  RType *Output() const { return output_; }

  const vector<int> &Strides() const { return strides_; }

  const vector<int> &Paddings() const { return paddings_; }

  const vector<int> &Dilations() const { return dilations_; }

  const int &Groups() const { return groups; }

 private:
  RType *input_;
  RType *output_;
  RType *filter_;
  vector<int> strides_;
  vector<int> paddings_;
  vector<int> dilations_;
  int groups;
};
#endif

}  // namespace operators
}  // namespace paddle_mobile
