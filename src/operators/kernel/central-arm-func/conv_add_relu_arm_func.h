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

#ifdef FUSION_CONVADDRELU_OP

#pragma once
#include <type_traits>
#include <vector>
#include "operators/math/conv_func.h"
#include "operators/math/im2col.h"
#include "operators/math/math_function.h"
#include "operators/math/vol2col.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename P, typename S>
void ConvAddReluCompute(const FusionConvAddReluParam<CPU> &param) {
  const Tensor *input = param.Input();
  Tensor filter = *param.Filter();
  Tensor bias = *param.Bias();
  int32_t axis = param.Axis();
  S *bias_data = bias.data<S>();
  Tensor *output = param.Output();
  output->mutable_data<P>();

  float alpha = 1.0f;
  float beta = 1.0f;

#ifdef FUSION_CONVADDRELU_INT8_OP
  if (std::is_same<P, int8_t>::value) {
    alpha = param.InputScale()->data<float>()[0];
    beta = 0.0f;
  }
#endif

  int32_t groups = param.Groups();
  std::vector<int32_t> strides = param.Strides();
  std::vector<int32_t> paddings = param.Paddings();
  std::vector<int32_t> dilations = param.Dilations();

  const int32_t batch_size = static_cast<int32_t>(input->dims()[0]);

  std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));

  std::vector<int64_t> output_shape_vec(framework::vectorize(output->dims()));
  size_t data_dim = filter_shape_vec.size() - 2;
  std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
  col_shape_vec[0] = input->dims()[1] / groups;
  for (size_t j = 0; j < data_dim; ++j) {
    col_shape_vec[j + 1] = filter_shape_vec[j + 2];
    col_shape_vec[j + 1 + data_dim] = output_shape_vec[j + 2];
  }
  framework::DDim col_shape(framework::make_ddim(col_shape_vec));

  framework::DDim col_matrix_shape =
      framework::flatten_to_2d(col_shape, data_dim + 1);

  bool is_expand =
      math::IsExpand(filter_shape_vec, strides, paddings, dilations);
  Tensor col;
  Tensor col_matrix;
  if (is_expand) {
    col.mutable_data<P>(col_shape);
    col_matrix.ShareDataWith(col);
    col_matrix.Resize(col_matrix_shape);
  }

  framework::DDim input_shape = framework::slice_ddim(
      input->dims(), 1, static_cast<int32_t>(input->dims().size()));

  framework::DDim filter_matrix_shape = {filter.dims()[0],
                                         filter.numel() / filter.dims()[0]};
  filter.Resize(filter_matrix_shape);
  framework::DDim output_matrix_shape = {
      output->dims()[1],
      output->numel() / (output->dims()[0] * output->dims()[1])};

  // convolution operator: im2col(or vol2col) + gemm
  int32_t in_step = static_cast<int32_t>(input->dims()[1]) / groups;
  int32_t out_step = static_cast<int32_t>(output->dims()[1]) / groups;

  math::Vol2ColFunctor<CPU, P> vol2col;
  math::Im2ColFunctor<math::ColFormat::kCFO, CPU, P> im2col;

  for (int32_t i = 0; i < batch_size; i++) {
    Tensor in_batch = input->Slice(i, i + 1).Resize(input_shape);
    Tensor out_batch = output->Slice(i, i + 1).Resize(output_matrix_shape);

    for (int32_t g = 0; g < groups; g++) {
      Tensor in_slice = in_batch.Slice(g * in_step, (g + 1) * in_step);

      if (!is_expand) {
        col.ShareDataWith(in_slice);
        col_matrix.ShareDataWith(col);
        col_matrix.Resize(col_matrix_shape);
      } else if (data_dim == 2U) {
        // im2col
        im2col(in_slice, dilations, strides,
               std::vector<int32_t>{paddings[0], paddings[1], paddings[0],
                                    paddings[1]},
               &col);
      } else if (data_dim == 3U) {
        // vol2col
        vol2col(in_slice, dilations, strides, paddings, &col);
      }

      // gemm
      Tensor out_slice = out_batch.Slice(g * out_step, (g + 1) * out_step);
      Tensor filter_slice = filter.Slice(g * out_step, (g + 1) * out_step);

      math::matmul(filter_slice, false, col_matrix, false, alpha, &out_slice,
                   beta, true, bias_data);
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
