/* Copyright (c) 2016 Baidu, Inc. All Rights Reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
==============================================================================*/

#include "operators/prior_box_op.h"
#include <vector>
namespace paddle_mobile {
namespace operators {

template <typename Dtype, typename T>
void PriorBoxOp<Dtype, T>::InferShape() const {
  auto input_dims = param_.Input()->dims();
  auto input_image_dims = param_.InputImage()->dims();
  auto min_sizes = param_.MinSizes();
  auto max_sizes = param_.MaxSizes();
  auto variances = param_.Variances();
  auto aspect_ratios = param_.AspectRatios();
  bool flip = param_.Flip();
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
  param_.OutputBoxes()->Resize(framework::make_ddim(dim_vec));
  param_.OutputVariances()->Resize(framework::make_ddim(dim_vec));
}
template class PriorBoxOp<CPU, float>;
}  // namespace operators
}  // namespace paddle_mobile
