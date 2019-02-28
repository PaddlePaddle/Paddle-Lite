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

#ifdef FUSION_DWCONVBNRELU_OP

#pragma once
#include <vector>
#include "operators/math/depthwise_conv3x3.h"
#include "operators/math/im2col.h"
#include "operators/math/math_function.h"
#include "operators/math/vol2col.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

void DWConvBNReluBasic(const FusionDWConvBNReluParam<CPU> &param) {
  const Tensor *input = param.Input();
  Tensor filter = *param.Filter();
  Tensor new_bias = *param.NewBias();
  Tensor new_scale = *param.NewScale();

  Tensor *output = param.Output();
  output->mutable_data<float>();

  int groups = param.Groups();
  std::vector<int> strides = param.Strides();
  std::vector<int> paddings = param.Paddings();
  std::vector<int> dilations = param.Dilations();

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
    col.mutable_data<float>(col_shape);
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

  math::Vol2ColFunctor<CPU, float> vol2col;
  math::Im2ColFunctor<math::ColFormat::kCFO, CPU, float> im2col;

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
      math::MatMulWithBn(filter_slice, false, col_matrix, false,
                         static_cast<float>(1), &out_slice,
                         static_cast<float>(0), true, &new_scale, &new_bias, g);
    }
  }
}
template <typename P>
void DWConvBNReluCompute(const FusionDWConvBNReluParam<CPU> &param) {
  if (param.Groups() == param.Input()->dims()[1] &&
      param.Input()->dims()[1] == param.Output()->dims()[1] &&
      param.Filter()->dims()[2] == param.Filter()->dims()[3] &&
      param.Filter()->dims()[2] == 3 && param.Strides()[0] == 1 &&
      param.paddings_[0] == 1) {
    math::DepthwiseConvAddBNRelu3x3s1p1(param.Input(), param.Filter(),
                                        param.Output(), param.NewScale(),
                                        param.NewBias(), true);
  } else if (param.Groups() == param.Input()->dims()[1] &&
             param.Input()->dims()[1] == param.Output()->dims()[1] &&
             param.Filter()->dims()[2] == param.Filter()->dims()[3] &&
             param.Filter()->dims()[2] == 3 && param.Strides()[0] == 2) {
    //    math::DepthwiseConvAddBNRelu3x3s2p1(param.Input(), param.Filter(),
    //                                        param.Output(), param.NewScale(),
    //                                        param.NewBias(), 1);
    math::DepthwiseConvAddBNRelu3x3s2p1v2(param.Input(), param.Filter(),
                                          param.Output(), param.NewScale(),
                                          param.NewBias(), true);
  } else {
    DWConvBNReluBasic(param);
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
