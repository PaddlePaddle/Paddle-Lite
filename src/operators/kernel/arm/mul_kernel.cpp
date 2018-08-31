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

#ifdef MUL_OP

#include "operators/kernel/mul_kernel.h"
#include "operators/kernel/central-arm-func/mul_arm_func.h"

namespace paddle_mobile {
namespace operators {

template <>
bool MulKernel<CPU, float>::Init(MulParam<CPU> *param) {
  return true;
}

template <>
void MulKernel<CPU, float>::Compute(const MulParam<CPU> &param) const {
  auto x_dims = param.InputX()->dims();
  auto y_dims = param.InputY()->dims();
  int x_num_col_dims = param.XNumColDims();
  int y_num_col_dims = param.YNumColDims();

  assert(x_dims.size() > x_num_col_dims);
  assert(y_dims.size() > y_num_col_dims);

  /// (1,2,3,4) , x_num_col_dims = 2  -> (2,12)
  auto x_mat_dims = framework::flatten_to_2d(x_dims, x_num_col_dims);
  auto y_mat_dims = framework::flatten_to_2d(y_dims, y_num_col_dims);

  assert(x_mat_dims[1] == y_mat_dims[0]);

  std::vector<int64_t> output_dims;
  output_dims.reserve(
          static_cast<size_t>(x_num_col_dims + y_dims.size() - y_num_col_dims));

  for (int i = 0; i < x_num_col_dims; ++i) {
    output_dims.push_back(x_dims[i]);
  }

  for (int i = y_num_col_dims; i < y_dims.size(); ++i) {
    output_dims.push_back(y_dims[i]);
  }

  framework::DDim ddim = framework::make_ddim(output_dims);
  param.Out()->Resize(ddim);
  MulCompute<float>(param);
  param.Out()->set_lod(param.InputX()->lod());
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
