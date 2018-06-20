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
#ifdef FUSION_CONVADD_OP

#include "operators/kernel/conv_add_kernel.h"

namespace paddle_mobile {
namespace operators {

void expand_bias(Tensor &bias, int axis, const DDim &dDim) {
  auto bias_ptr = bias.data<float>();
  const DDim bias_ddim = bias.dims();
  PADDLE_MOBILE_ENFORCE(bias.dims().size() == 1,
                        "the bias tensor's dims size != 1")
  DDim outer_ddim = paddle_mobile::framework::slice_ddim(dDim, 0, axis + 1);
  DDim inner_ddim =
      paddle_mobile::framework::slice_ddim(dDim, axis + 1, dDim.size());
  int outer_size = paddle_mobile::framework::product(outer_ddim);
  int inner_size = paddle_mobile::framework::product(inner_ddim);
  bias.Resize(dDim);
  auto new_ptr = bias.mutable_data<float>();
  int axis_size = dDim[axis];
  for (int i = 0; i < outer_size; ++i) {
    float v_bias = bias_ptr[i * axis_size / outer_size];
    for (int j = 0; j < inner_size; ++j) {
      new_ptr[i * inner_size + j] = v_bias;
    }
  }
}

template <>
void ConvAddKernel<CPU, float>::Compute(
    const FushionConvAddParam &param) const {
  const Tensor *input = param.Input();
  Tensor filter = *param.Filter();
  Tensor bias = *param.Bias();
  int axis = param.Axis();
  Tensor *output = param.Output();
  expand_bias(bias, axis, output->dims());
  output->ShareDataWith(bias);
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

  bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);
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
      math::matmul<float>(filter_slice, false, col_matrix, false,
                          static_cast<float>(1), &out_slice,
                          static_cast<float>(1));
    }
  }
}
template class ConvAddKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
