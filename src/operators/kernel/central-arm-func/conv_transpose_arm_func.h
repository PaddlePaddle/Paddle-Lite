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

#ifdef CONV_TRANSPOSE_OP

#include <vector>
#include "framework/ddim.h"
#include "operators/math/im2col.h"
#include "operators/math/math_function.h"
#include "operators/math/vol2col.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename P>
void ConvTransposeCompute(const ConvTransposeParam<CPU> &param) {
  const Tensor *input = param.Input();
  Tensor filter = *param.Filter();
  Tensor *output = param.Output();
  output->mutable_data<P>();

  auto strides = param.Strides();
  auto paddings = param.Paddings();
  auto dilations = param.Dilations();
  auto groups = param.Groups();

  const int batch_size = input->dims()[0];

  std::vector<int64_t> input_shape_vec = framework::vectorize(input->dims());
  std::vector<int64_t> filter_shape_vec = framework::vectorize(filter.dims());

  size_t data_dim = filter_shape_vec.size() - 2;

  // 5 或者 7
  std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);

  // output c / groups
  col_shape_vec[0] = output->dims()[1] / groups;
  for (size_t i = 0; i < data_dim; ++i) {
    // filter shape  filter h  filter w
    col_shape_vec[i + 1] = filter_shape_vec[i + 2];
    // input shape  input h  input w
    col_shape_vec[i + 1 + data_dim] = input_shape_vec[i + 2];
  }

  framework::DDim col_shape(framework::make_ddim(col_shape_vec));
  framework::DDim col_matrix_shape =
      framework::flatten_to_2d(col_shape, data_dim + 1);

  Tensor col;
  col.mutable_data<P>(col_shape);

  Tensor col_matrix;
  col_matrix.ShareDataWith(col);
  col_matrix.Resize(col_matrix_shape);

  framework::DDim output_shape =
      framework::slice_ddim(output->dims(), 1, output->dims().size());

  framework::DDim input_matrix_shape = {input->dims()[1], col_matrix_shape[1]};

  // filter size: (m, c/g * k_h * k_w) or (m, c/g * k_d * k_h * k_w)
  framework::DDim filter_matrix_shape = {input->dims()[1], col_matrix_shape[0]};
  filter.Resize(filter_matrix_shape);

  int in_step = static_cast<int>(input->dims()[1]) / groups;
  int out_step = static_cast<int>(output->dims()[1]) / groups;

  math::Col2ImFunctor<math::ColFormat::kCFO, CPU, P> col2im;
  math::Col2VolFunctor<CPU, P> col2vol;

  for (int i = 0; i < batch_size; ++i) {
    Tensor input_batch = input->Slice(i, i + 1).Resize(input_matrix_shape);
    Tensor output_batch = output->Slice(i, i + 1).Resize(output_shape);

    for (int g = 0; g < groups; ++g) {
      Tensor in_slice = input_batch.Slice(g * in_step, (g + 1) * in_step);
      Tensor filter_slice = filter.Slice(g * in_step, (g + 1) * in_step);
      Tensor out_slice = output_batch.Slice(g * out_step, (g + 1) * out_step);

      math::MatMul<P, P>(filter_slice, true, in_slice, false,
                         static_cast<P>(1.0), &col_matrix, static_cast<P>(0.0));
      if (data_dim == 2U) {
        col2im(col, dilations, strides,
               std::vector<int>{paddings[0], paddings[1], paddings[0],
                                paddings[1]},
               &out_slice);
      } else if (data_dim == 3U) {
        col2vol(col, dilations, strides, paddings, &out_slice);
      }
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile
#endif
