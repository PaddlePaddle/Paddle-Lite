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

#ifdef CONV_OP

#pragma once
#include <vector>
#include "operators/math/conv_func.h"
#include "operators/math/depthwise_conv_3x3.h"
#include "operators/math/im2col.h"
#include "operators/math/math_function.h"
#include "operators/math/pad.h"
#include "operators/math/vol2col.h"
#include "operators/math/winograd/winograd.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename Itype, typename Otype>
inline void ConvBasic(const ConvParam<CPU> &param) {
  const Tensor *input = param.Input();
  Tensor filter = *param.Filter();
  Tensor *output = param.Output();
  output->mutable_data<Otype>();
  int groups = param.Groups();
  const std::vector<int> strides = param.Strides();
  const std::vector<int> paddings = param.Paddings();
  const std::vector<int> dilations = param.Dilations();

  const int batch_size = static_cast<int>(input->dims()[0]);

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
    col.mutable_data<Itype>(col_shape);
    col_matrix.ShareDataWith(col);
    col_matrix.Resize(col_matrix_shape);
  }

  framework::DDim input_shape = framework::slice_ddim(
      input->dims(), 1, static_cast<int>(input->dims().size()));

  framework::DDim filter_matrix_shape = {filter.dims()[0],
                                         filter.numel() / filter.dims()[0]};
  filter.Resize(filter_matrix_shape);
  framework::DDim output_matrix_shape = {
      output->dims()[1],
      output->numel() / (output->dims()[0] * output->dims()[1])};

  // convolution operator: im2col(or vol2col) + gemm
  int in_step = static_cast<int>(input->dims()[1]) / groups;
  int out_step = static_cast<int>(output->dims()[1]) / groups;

  math::Vol2ColFunctor<CPU, Itype> vol2col;
  math::Im2ColFunctor<math::ColFormat::kCFO, CPU, Itype> im2col;

  for (int i = 0; i < batch_size; i++) {
    Tensor in_batch = input->Slice(i, i + 1).Resize(input_shape);
    Tensor out_batch = output->Slice(i, i + 1).Resize(output_matrix_shape);

    for (int g = 0; g < groups; g++) {
      Tensor in_slice = in_batch.Slice(g * in_step, (g + 1) * in_step);

      if (!is_expand) {
        col.ShareDataWith(in_slice);
        col_matrix.ShareDataWith(col);
        col_matrix.Resize(col_matrix_shape);
      } else if (data_dim == 2U) {
        // im2col
        im2col(in_slice, dilations, strides,
               std::vector<int>{paddings[0], paddings[1], paddings[0],
                                paddings[1]},
               &col);

      } else if (data_dim == 3U) {
        // vol2col
        vol2col(in_slice, dilations, strides, paddings, &col);
      }

      // gemm
      Tensor out_slice = out_batch.Slice(g * out_step, (g + 1) * out_step);
      Tensor filter_slice = filter.Slice(g * out_step, (g + 1) * out_step);

      math::matmul<Itype>(filter_slice, false, col_matrix, false,
                          static_cast<float>(1), &out_slice,
                          static_cast<float>(0));
    }
  }
}

inline void BatchConv3x3Winograd(const ConvParam<CPU> &param) {
  const Tensor *input = param.Input();
  Tensor *filter = param.Filter();
  Tensor *output = param.Output();
  output->mutable_data<float>();
  int batch_size = input->dims()[0];
  int groups = param.Groups();
  const std::vector<int> &paddings = param.Paddings();
  math::PadFunctor<CPU, float> pad;

  Tensor input_pad;
  for (int i = 0; i < batch_size; ++i) {
    Tensor in_batch = input->Slice(i, i + 1);
    Tensor out_batch = output->Slice(i, i + 1);
    if (paddings[0] == 0 && paddings[1] == 0) {
      input_pad = in_batch;
    } else {
      framework::DDim pad_shape = in_batch.dims();
      pad_shape[2] += 2 * paddings[0];
      pad_shape[3] += 2 * paddings[1];
      input_pad.mutable_data<float>(pad_shape);
      pad(in_batch, paddings[0], paddings[0], paddings[1], paddings[1],
          &input_pad);
    }
    math::winograd_f6k3(input_pad, *filter, &out_batch);
  }
}

template <typename P>
void ConvCompute(const ConvParam<CPU> &param) {
  if (param.Input()->type() == typeid(int8_t)) {
    ConvBasic<int8_t, int32_t>(param);
  } else {
    if (param.Groups() == param.Input()->dims()[1] &&
        param.Input()->dims()[1] == param.Output()->dims()[1] &&
        param.Filter()->dims()[2] == param.Filter()->dims()[3] &&
        param.Filter()->dims()[2] == 3 && param.Strides()[0] == 1) {
      math::DepthwiseConv3x3s1p1(param.Input(), param.Filter(), param.Output(),
                                 nullptr, false);
    } else if (param.Groups() == param.Input()->dims()[1] &&
               param.Input()->dims()[1] == param.Output()->dims()[1] &&
               param.Filter()->dims()[2] == param.Filter()->dims()[3] &&
               param.Filter()->dims()[2] == 3) {
      math::DepthwiseConv3x3(param.Input(), param.Strides(), param.Paddings(),
                             param.Filter(), nullptr, param.Output(), false);
    } else if (param.Filter()->dims()[2] == param.Filter()->dims()[3] &&
               param.Strides()[0] == param.Strides()[1] &&
               param.Dilations()[0] == param.Dilations()[1] &&
               param.Filter()->dims()[2] == 3 && param.Strides()[0] == 1 &&
               param.Dilations()[0] == 1 && param.Input()->dims()[1] > 16) {
      BatchConv3x3Winograd(param);
    } else {
      ConvBasic<float, float>(param);
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
