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

#include "operators/prior_box_op.h"
#include <vector>

namespace paddle_mobile {
namespace operators {

#ifdef PRIORBOX_OP
template <typename Dtype, typename T>
void PriorBoxOp<Dtype, T>::InferShape() const {
  auto input_dims = this->param_.Input()->dims();
  auto input_image_dims = this->param_.InputImage()->dims();
  auto min_sizes = this->param_.MinSizes();
  auto max_sizes = this->param_.MaxSizes();
  auto variances = this->param_.Variances();
  auto aspect_ratios = this->param_.AspectRatios();
  bool flip = this->param_.Flip();
  std::vector<float> aspect_ratios_vec;
  ExpandAspectRatios(aspect_ratios, flip, &aspect_ratios_vec);

  size_t num_priors = aspect_ratios_vec.size() * min_sizes.size();
  if (!max_sizes.empty()) {
    num_priors += max_sizes.size();
  }

  std::vector<int64_t> dim_vec(4);
  dim_vec[0] = input_dims[2];
  dim_vec[1] = input_dims[3];
  dim_vec[2] = num_priors;
  dim_vec[3] = 4;
  this->param_.OutputBoxes()->Resize(framework::make_ddim(dim_vec));
  this->param_.OutputVariances()->Resize(framework::make_ddim(dim_vec));
}
#endif  // PRIORBOX_OP

#ifdef DENSITY_PRIORBOX_OP
template <typename Dtype, typename T>
void DensityPriorBoxOp<Dtype, T>::InferShape() const {
  auto input_dims = this->param_.Input()->dims();
  auto input_image_dims = this->param_.InputImage()->dims();

  auto &fixed_sizes = this->param_.FixedSizes();
  auto &fixed_ratios = this->param_.FixedRatios();
  auto &densities = this->param_.Densities();
  bool flatten = this->param_.FlattenTo2d();

  size_t num_priors = 0;
  for (size_t i = 0; i < densities.size(); ++i) {
    num_priors += (fixed_ratios.size()) * (pow(densities[i], 2));
  }
  if (!flatten) {
    std::vector<int64_t> dim_vec(4);
    dim_vec[0] = input_dims[2];
    dim_vec[1] = input_dims[3];
    dim_vec[2] = num_priors;
    dim_vec[3] = 4;
    this->param_.OutputBoxes()->Resize(framework::make_ddim(dim_vec));
    this->param_.OutputVariances()->Resize(framework::make_ddim(dim_vec));
  } else {
    int64_t dim0 = input_dims[2] * input_dims[3] * num_priors;
    this->param_.OutputBoxes()->Resize(framework::make_ddim({dim0, 4}));
    this->param_.OutputVariances()->Resize(framework::make_ddim({dim0, 4}));
  }
}
#endif  // DENSITY_PRIORBOX_OP

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;

#ifdef PADDLE_MOBILE_CPU
#ifdef PRIORBOX_OP
REGISTER_OPERATOR_CPU(prior_box, ops::PriorBoxOp);
#endif  // PRIORBOX_OP
#ifdef DENSITY_PRIORBOX_OP
REGISTER_OPERATOR_CPU(density_prior_box, ops::DensityPriorBoxOp);
#endif  // DENSITY_PRIORBOX_OP
#endif  // PADDLE_MOBILE_CPU

#ifdef PADDLE_MOBILE_CL
#ifdef PRIORBOX_OP
REGISTER_OPERATOR_CL(prior_box, ops::PriorBoxOp);
#endif  // PRIORBOX_OP
#endif  // PADDLE_MOBILE_CL

#if defined(PADDLE_MOBILE_FPGA) || defined(PADDLE_MOBILE_FPGA_KD)
REGISTER_OPERATOR_FPGA(prior_box, ops::PriorBoxOp);
#endif
