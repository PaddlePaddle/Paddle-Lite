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

#ifdef FUSION_CONVADDADDPRELU_OP

#pragma once
#include <vector>
#include "operators/math/conv_func.h"
#include "operators/math/im2col.h"
#include "operators/math/math_function.h"
#include "operators/math/vol2col.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

template <typename P>
void ConvAddAddPReluCompute(const FusionConvAddAddPReluParam<CPU> &param) {
  const Tensor *input = param.Input();
  Tensor filter = *param.Filter();
  Tensor bias = *param.Bias();
  Tensor bias1 = *param.Bias1();
  int axis = param.Axis();
  Tensor *output = param.Output();

  float *biase_data = bias.data<float>();

  int groups = param.Groups();
  std::vector<int> strides = param.Strides();
  std::vector<int> paddings = param.Paddings();
  std::vector<int> dilations = param.Dilations();
  Tensor aa = *param.InputAlpha();
  float *p = aa.data<float>();

  std::string mode = param.Mode();
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
    Tensor bias1_batch = bias1.Slice(i, i + 1).Resize(output_matrix_shape);
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
      Tensor bias1_slice = bias1_batch.Slice(g * out_step, (g + 1) * out_step);
      float *biase_data1 = bias1_slice.data<float>();
      //                    int n = bias1_slice.dims()[0];
      //                    int m = bias1_slice.dims()[1];
      //                    for(int i=0;i<n*m;i++){
      //                        if(biase_data1[i]!=0)
      //                        DLOG<<biase_data1[i]<<",yangfei";
      //                    }

      //                    math::matmul<float>(filter_slice, false, col_matrix,
      //                    false,
      //                                        static_cast<float>(1),
      //                                        &out_slice,
      //                                        static_cast<float>(1), true,
      //                                        biase_data);
      math::matmulWithPRelu(filter_slice, false, col_matrix, false, &out_slice,
                            p, mode, biase_data, biase_data1);
    }
  }
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
