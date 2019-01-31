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

#include <vector>
#include "framework/operator.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

#ifdef ANCHOR_GENERATOR_OP
template <typename Dtype>
class AnchorGeneratorParam : public OpParam {
 public:
  AnchorGeneratorParam(const VariableNameMap &inputs,
                       const VariableNameMap &outputs,
                       const AttributeMap &attrs, const Scope &scope) {
    input_ = OpParam::GetVarValue<framework::LoDTensor>("Input", inputs, scope);
    output_anchors_ =
        OpParam::GetVarValue<framework::LoDTensor>("Anchors", outputs, scope);
    output_variances_ =
        OpParam::GetVarValue<framework::LoDTensor>("Variances", outputs, scope);

    anchor_sizes_ = OpParam::GetAttr<std::vector<float>>("anchor_sizes", attrs);
    aspect_ratios_ =
        OpParam::GetAttr<std::vector<float>>("aspect_ratios", attrs);
    variances_ = OpParam::GetAttr<std::vector<float>>("variances", attrs);
    stride_ = OpParam::GetAttr<std::vector<float>>("stride", attrs);
    offset_ = OpParam::GetAttr<float>("offset", attrs);
  }

 public:
  // input
  framework::Tensor *input_;
  // outputs
  framework::Tensor *output_anchors_;
  framework::Tensor *output_variances_;

  std::vector<float> anchor_sizes_;
  std::vector<float> aspect_ratios_;
  std::vector<float> variances_;
  std::vector<float> stride_;
  float offset_;
};

DECLARE_KERNEL(AnchorGenerator, AnchorGeneratorParam);
#endif

#ifdef PROPOSAL_OP
template <typename Dtype>
class ProposalParam : public OpParam {
 public:
  ProposalParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
                const AttributeMap &attrs, const Scope &scope) {
    scores_ =
        OpParam::GetVarValue<framework::LoDTensor>("Scores", inputs, scope);
    bbox_deltas_ =
        OpParam::GetVarValue<framework::LoDTensor>("BboxDeltas", inputs, scope);
    im_info_ =
        OpParam::GetVarValue<framework::LoDTensor>("ImInfo", inputs, scope);
    anchors_ =
        OpParam::GetVarValue<framework::LoDTensor>("Anchors", inputs, scope);
    variances_ =
        OpParam::GetVarValue<framework::LoDTensor>("Variances", inputs, scope);

    rpn_rois_ =
        OpParam::GetVarValue<framework::LoDTensor>("RpnRois", outputs, scope);
    rpn_probs_ = OpParam::GetVarValue<framework::LoDTensor>("RpnRoiProbs",
                                                            outputs, scope);

    pre_nms_topn_ = OpParam::GetAttr<int>("pre_nms_topN", attrs);
    post_nms_topn_ = OpParam::GetAttr<int>("post_nms_topN", attrs);
    nms_thresh_ = OpParam::GetAttr<float>("nms_thresh", attrs);
    min_size_ = OpParam::GetAttr<float>("min_size", attrs);
    eta_ = OpParam::GetAttr<float>("eta", attrs);
  }

 public:
  framework::Tensor *scores_;
  framework::Tensor *bbox_deltas_;
  framework::Tensor *im_info_;
  framework::Tensor *anchors_;
  framework::Tensor *variances_;

  framework::LoDTensor *rpn_rois_;
  framework::LoDTensor *rpn_probs_;

  int pre_nms_topn_;
  int post_nms_topn_;
  float nms_thresh_;
  float min_size_;
  float eta_;
};

DECLARE_KERNEL(Proposal, ProposalParam);
#endif

#ifdef PSROI_POOL_OP
template <typename Dtype>
class PSRoiPoolParam : public OpParam {
 public:
  PSRoiPoolParam(const VariableNameMap &inputs, const VariableNameMap &outputs,
                 const AttributeMap &attrs, const Scope &scope) {
    input_x_ = OpParam::GetVarValue<framework::LoDTensor>("X", inputs, scope);
    input_rois_ =
        OpParam::GetVarValue<framework::LoDTensor>("ROIs", inputs, scope);
    output_ = OpParam::GetVarValue<framework::LoDTensor>("Out", outputs, scope);

    output_channels_ = OpParam::GetAttr<int>("output_channels", attrs);
    pooled_height_ = OpParam::GetAttr<int>("pooled_height", attrs);
    pooled_width_ = OpParam::GetAttr<int>("pooled_width", attrs);
    spatial_scale_ = OpParam::GetAttr<float>("spatial_scale", attrs);
  }

 public:
  framework::Tensor *input_x_;
  framework::LoDTensor *input_rois_;
  framework::Tensor *output_;
  int output_channels_;
  int pooled_height_;
  int pooled_width_;
  float spatial_scale_;
};

DECLARE_KERNEL(PSRoiPool, PSRoiPoolParam);
#endif

#ifdef ROI_PERSPECTIVE_OP
template <typename Dtype>
class RoiPerspectiveParam : public OpParam {
 public:
  RoiPerspectiveParam(const VariableNameMap &inputs,
                      const VariableNameMap &outputs, const AttributeMap &attrs,
                      const Scope &scope) {
    input_x_ = OpParam::GetVarValue<framework::LoDTensor>("X", inputs, scope);
    input_rois_ =
        OpParam::GetVarValue<framework::LoDTensor>("ROIs", inputs, scope);
    output_ = OpParam::GetVarValue<framework::LoDTensor>("Out", outputs, scope);

    spatial_scale_ = OpParam::GetAttr<float>("spatial_scale", attrs);
    transformed_height_ = OpParam::GetAttr<int>("transformed_height", attrs);
    transformed_width_ = OpParam::GetAttr<int>("transformed_width", attrs);
  }

 public:
  framework::Tensor *input_x_;
  framework::LoDTensor *input_rois_;
  framework::Tensor *output_;

  float spatial_scale_;
  int transformed_height_;
  int transformed_width_;
};

DECLARE_KERNEL(RoiPerspective, RoiPerspectiveParam);
#endif

}  // namespace operators
}  // namespace paddle_mobile
