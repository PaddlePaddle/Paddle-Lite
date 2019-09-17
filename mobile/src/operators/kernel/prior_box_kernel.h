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

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>
#include "framework/operator.h"
#include "operators/math/transform.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

#ifdef PRIORBOX_OP
inline void ExpandAspectRatios(const std::vector<float> &input_aspect_ratior,
                               bool flip,
                               std::vector<float> *output_aspect_ratior) {
  constexpr float epsilon = 1e-6;
  output_aspect_ratior->clear();
  output_aspect_ratior->push_back(1.0f);
  for (size_t i = 0; i < input_aspect_ratior.size(); ++i) {
    float ar = input_aspect_ratior[i];
    bool already_exist = false;
    for (size_t j = 0; j < output_aspect_ratior->size(); ++j) {
      if (fabs(ar - output_aspect_ratior->at(j)) < epsilon) {
        already_exist = true;
        break;
      }
    }
    if (!already_exist) {
      output_aspect_ratior->push_back(ar);
      if (flip) {
        output_aspect_ratior->push_back(1.0f / ar);
      }
    }
  }
}

DECLARE_KERNEL(PriorBox, PriorBoxParam);
#endif  // PRIORBOX_OP

#ifdef DENSITY_PRIORBOX_OP
template <typename Dtype>
class DensityPriorBoxParam : public OpParam {
  typedef typename DtypeTensorTrait<Dtype>::gtype GType;

 public:
  DensityPriorBoxParam(const VariableNameMap &inputs,
                       const VariableNameMap &outputs,
                       const AttributeMap &attrs, Scope *scope)
      : OpParam(inputs, outputs, attrs, scope) {
    input_ = InputFrom<GType>(inputs, *scope);
    input_image_ = InputImageFrom<GType>(inputs, *scope);
    output_boxes_ = OutputBoxesFrom<GType>(outputs, *scope);
    output_variances_ = OutputVariancesFrom<GType>(outputs, *scope);
    variances_ = GetAttr<vector<float>>("variances", attrs);
    clip_ = GetAttr<bool>("clip", attrs);
    flatten_to_2d_ = GetAttr<bool>("flatten_to_2d", attrs);
    step_w_ = GetAttr<float>("step_w", attrs);
    step_h_ = GetAttr<float>("step_h", attrs);
    offset_ = GetAttr<float>("offset", attrs);
    fixed_sizes_ = GetAttr<vector<float>>("fixed_sizes", attrs);
    fixed_ratios_ = GetAttr<vector<float>>("fixed_ratios", attrs);
    densities_ = GetAttr<vector<int>>("densities", attrs);
  }

  ~DensityPriorBoxParam() {}

  const GType *Input() const { return input_; }
  const GType *InputImage() const { return input_image_; }
  GType *OutputBoxes() const { return output_boxes_; }
  GType *OutputVariances() const { return output_variances_; }
  const bool Clip() const { return clip_; }
  const bool FlattenTo2d() const { return flatten_to_2d_; }
  const float StepW() const { return step_w_; }
  const float StepH() const { return step_h_; }
  const float Offset() const { return offset_; }
  const vector<float> &FixedSizes() const { return fixed_sizes_; }
  const vector<float> &FixedRatios() const { return fixed_ratios_; }
  const vector<int> &Densities() const { return densities_; }
  const vector<float> &Variances() const { return variances_; }
  GType *getNewDensity() const { return new_density.get(); }
  void setNewDensity(GType *newDensity) { new_density.reset(newDensity); }

 public:
  GType *input_;
  GType *input_image_;
  GType *output_boxes_;
  GType *output_variances_;
  bool clip_;
  bool flatten_to_2d_;
  float step_w_;
  float step_h_;
  float offset_;
  vector<float> fixed_sizes_;
  vector<float> fixed_ratios_;
  vector<int> densities_;
  vector<float> variances_;
  std::shared_ptr<GType> new_density;
};

DECLARE_KERNEL(DensityPriorBox, DensityPriorBoxParam);
#endif  // DENSITY_PRIORBOX_OP

}  // namespace operators
}  // namespace paddle_mobile
