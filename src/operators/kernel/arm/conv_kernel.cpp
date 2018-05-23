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

#include "operators/kernel/conv_kernel.h"

namespace paddle_mobile {
namespace operators {

bool IsExpand(const std::vector<int64_t> &filter_dim,
              const std::vector<int> &strides, const std::vector<int> &paddings,
              const std::vector<int> &dilations) {
  bool filter_1 = true, strides_1 = true, padding_0 = true, dilation_1 = true;
  for (size_t j = 0; j < strides.size(); ++j) {
    filter_1 = filter_1 && (static_cast<int>(filter_dim[j + 2]) == 1);
    strides_1 = strides_1 && (strides[j] == 1);
    padding_0 = padding_0 && (paddings[j] == 0);
    dilation_1 = dilation_1 && (dilations[j] == 1);
  }
  return !(filter_1 && strides_1 && padding_0 && dilation_1);
}

template <>
void ConvKernel<CPU, float>::Compute(const ConvParam &param) const {
  LOG(kLOG_DEBUG) << param;

  const Tensor *input = param.Input();

  // The filter will be reshaped in the calculations,
  // so here use an assignment operation,
  // that avoids modifying the variable in the Scope.
  Tensor filter = *param.Filter();

  Tensor *output = param.Output();
  //            output->mutable_data<T>(context.GetPlace());

  int groups = param.Groups();
  std::vector<int> strides = param.Strides();
  std::vector<int> paddings = param.Paddings();
  std::vector<int> dilations = param.Dilations();

  DLOG << " compute end get Attrs " << strides[0];

  const int batch_size = static_cast<int>(input->dims()[0]);

  // filter_shape_vec: {k_o, k_i, k_h, k_w} or {k_o, k_i, k_d, k_h,
  // k_w}
  std::vector<int64_t> filter_shape_vec(framework::vectorize(filter.dims()));
  // output_shape_vec: {o_n, o_c, o_h, o_w} or {o_n, o_c, o_d, o_h,
  // o_w}
  std::vector<int64_t> output_shape_vec(framework::vectorize(output->dims()));

  // use col_shape in the im2col calculation
  // col_shape_vec: {i_c/g, k_h, k_w, o_h, o_w} or {i_c/g, k_d, k_h,
  // k_w, o_d,
  // o_h, o_w}
  size_t data_dim = filter_shape_vec.size() - 2;
  std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
  col_shape_vec[0] = input->dims()[1] / groups;
  for (size_t j = 0; j < data_dim; ++j) {
    col_shape_vec[j + 1] = filter_shape_vec[j + 2];
    col_shape_vec[j + 1 + data_dim] = output_shape_vec[j + 2];
  }
  framework::DDim col_shape(framework::make_ddim(col_shape_vec));

  // use col_matrix_shape in the gemm calculation
  // size: (i_c/g * k_h * k_w, o_h * o_w) or (i_c/g * k_d * k_h * k_w,
  // o_d *
  // o_h * o_w)
  framework::DDim col_matrix_shape =
      framework::flatten_to_2d(col_shape, data_dim + 1);

  bool is_expand = IsExpand(filter_shape_vec, strides, paddings, dilations);
  Tensor col;
  // col_matrix shares the same piece of data with col,
  // but will be reshaped into a two-dimensional matrix shape
  // to call the matrix multiplication interface.
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

  //            auto& dev_ctx = context.template
  //            device_context<DeviceContext>();
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
                          static_cast<float>(0));
    }
  }
}

template class ConvKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile
