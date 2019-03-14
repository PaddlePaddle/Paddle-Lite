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

#ifdef PRIORBOX_OP

#include "operators/prior_box_op.h"
#include <vector>
namespace paddle_mobile {
namespace operators {

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

}  // namespace operators
}  // namespace paddle_mobile

namespace ops = paddle_mobile::operators;
#ifdef PADDLE_MOBILE_CPU
REGISTER_OPERATOR_CPU(prior_box, ops::PriorBoxOp);
#endif
#ifdef PADDLE_MOBILE_CL
REGISTER_OPERATOR_CL(prior_box, ops::PriorBoxOp);
#endif
#endif
