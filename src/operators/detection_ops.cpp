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

#include "operators/detection_ops.h"
#include <vector>

namespace paddle_mobile {
namespace operators {

#ifdef ANCHOR_GENERATOR_OP
template <typename DeviceType, typename T>
void AnchorGeneratorOp<DeviceType, T>::InferShape() const {
  const auto &input_dims = this->param_.input_->dims();
  // DLOG << "AnchorGenerator input dim =" << input_dims.size();
  PADDLE_MOBILE_ENFORCE(input_dims.size() == 4, "The layout of input is NCHW.");
  const auto &anchor_sizes = this->param_.anchor_sizes_;
  const auto &aspect_ratios = this->param_.aspect_ratios_;

  size_t num_anchors = aspect_ratios.size() * anchor_sizes.size();
  std::vector<int64_t> dim_vec(4);
  dim_vec[0] = input_dims[2];
  dim_vec[1] = input_dims[3];
  dim_vec[2] = num_anchors;
  dim_vec[3] = 4;

  this->param_.output_anchors_->Resize(framework::make_ddim(dim_vec));
  this->param_.output_variances_->Resize(framework::make_ddim(dim_vec));
}
#endif

#ifdef PROPOSAL_OP
template <typename DeviceType, typename T>
void ProposalOp<DeviceType, T>::InferShape() const {
  this->param_.rpn_rois_->Resize(framework::make_ddim({-1, 4}));
  this->param_.rpn_probs_->Resize(framework::make_ddim({-1, 1}));
}
#endif

#ifdef PSROI_POOL_OP
template <typename DeviceType, typename T>
void PSRoiPoolOp<DeviceType, T>::InferShape() const {
  const auto &rois_dims = this->param_.input_rois_->dims();
  const int pooled_height = this->param_.pooled_height_;
  const int pooled_width = this->param_.pooled_width_;
  const int output_channels = this->param_.output_channels_;

  auto out_dims = this->param_.input_x_->dims();
  out_dims[0] = rois_dims[0];
  out_dims[1] =
      output_channels;  // input_dims[1] / (pooled_height * pooled_width);
  out_dims[2] = pooled_height;
  out_dims[3] = pooled_width;
  this->param_.output_->Resize(out_dims);
}
#endif

#ifdef ROIALIGN_POOL_OP
template <typename DeviceType, typename T>
void RoiAlignPoolOp<DeviceType, T>::InferShape() const {
  const auto &rois_dims = this->param_.input_rois_->dims();
  const int pooled_height = this->param_.pooled_height_;
  const int pooled_width = this->param_.pooled_width_;

  auto out_dims = this->param_.input_x_->dims();
  out_dims[0] = rois_dims[0];
  // out_dims[1] =
  //     output_channels;  // input_dims[1] / (pooled_height * pooled_width);
  out_dims[2] = pooled_height;
  out_dims[3] = pooled_width;
  this->param_.output_->Resize(out_dims);
}
#endif

#ifdef ROI_PERSPECTIVE_OP
template <typename DeviceType, typename T>
void RoiPerspectiveOp<DeviceType, T>::InferShape() const {
  const auto &input_dims = this->param_.input_x_->dims();
  const auto &rois_dims = this->param_.input_rois_->dims();
  const int transformed_height = this->param_.transformed_height_;
  const int transformed_width = this->param_.transformed_width_;
  std::vector<int64_t> out_dims_v({rois_dims[0],   // num_rois
                                   input_dims[1],  // channels
                                   static_cast<int64_t>(transformed_height),
                                   static_cast<int64_t>(transformed_width)});
  auto out_dims = framework::make_ddim(out_dims_v);
  this->param_.output_->Resize(out_dims);
}
#endif

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
#ifdef ANCHOR_GENERATOR_OP
REGISTER_OPERATOR_CPU(anchor_generator, ops::AnchorGeneratorOp);
#endif
#ifdef PROPOSAL_OP
REGISTER_OPERATOR_CPU(generate_proposals, ops::ProposalOp);
#endif
#ifdef PSROI_POOL_OP
REGISTER_OPERATOR_CPU(psroi_pool, ops::PSRoiPoolOp);
#endif
#ifdef ROI_PERSPECTIVE_OP
REGISTER_OPERATOR_CPU(roi_perspective_transform, ops::RoiPerspectiveOp);
#endif
#endif

#ifdef PADDLE_MOBILE_FPGA
#ifdef ANCHOR_GENERATOR_OP
REGISTER_OPERATOR_FPGA(anchor_generator, ops::AnchorGeneratorOp);
#endif
#ifdef PROPOSAL_OP
REGISTER_OPERATOR_FPGA(generate_proposals, ops::ProposalOp);
#endif
#ifdef PSROI_POOL_OP
REGISTER_OPERATOR_FPGA(psroi_pool, ops::PSRoiPoolOp);
#endif
#ifdef ROIALIGN_POOL_OP
REGISTER_OPERATOR_FPGA(roialign_pool, ops::RoiAlignPoolOp);
#endif

#endif
