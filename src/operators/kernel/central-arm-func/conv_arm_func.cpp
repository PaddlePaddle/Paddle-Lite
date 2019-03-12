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

#include "operators/kernel/central-arm-func/conv_arm_func.h"
#include <vector>
#include "operators/math/depthwise_conv3x3.h"
#include "operators/math/depthwise_conv5x5.h"
#include "operators/math/im2col.h"
#include "operators/math/math_function.h"
#include "operators/math/pad.h"
#include "operators/math/vol2col.h"
#include "operators/math/winograd/winograd_transform.h"
#include "operators/op_param.h"

namespace paddle_mobile {
namespace operators {

int ConvOutputSize(int input_size, int filter_size, int dilation, int padding,
                   int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + 2 * padding - dkernel) / stride + 1;
  return output_size;
}

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

template <typename Itype, typename Otype>
void GemmConv(const ConvParam<CPU> &param) {
  const Tensor *input = param.Input();
  Tensor filter = *param.Filter();
  Tensor *output = param.Output();
  output->mutable_data<Otype>();

  int groups = param.Groups();
  const std::vector<int> strides = param.Strides();
  const std::vector<int> paddings = param.Paddings();
  const std::vector<int> dilations = param.Dilations();

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

  const int batch_size = static_cast<int>(input->dims()[0]);
  for (int i = 0; i < batch_size; i++) {
    Tensor in_batch = input->Slice(i, i + 1).Resize(input_shape);
    Tensor out_batch = output->Slice(i, i + 1).Resize(output_matrix_shape);

    for (int g = 0; g < groups; g++) {
      Tensor in_slice = in_batch.Slice(g * in_step, (g + 1) * in_step);

      if (!is_expand) {
        // col_matrix.ShareDataWith(in_slice);
        col_matrix = in_slice;
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

      math::MatMul<Itype, Otype>(filter_slice, false, col_matrix, false,
                                 static_cast<float>(1), &out_slice,
                                 static_cast<float>(0), false,
                                 static_cast<Otype *>(nullptr));
    }
  }
}

template <int tile, int kernel>
void WinogradConv3x3(const ConvParam<CPU> &param) {
  const Tensor *input = param.Input();
  const Tensor *filter = param.transformed_filter_;
  Tensor *output = param.Output();
  output->mutable_data<float>();
  int batch_size = input->dims()[0];
  int groups = param.Groups();
  const std::vector<int> &paddings = param.Paddings();

  auto winograd_pad = [&](int width, int pad) {
    int output_tile = tile - kernel + 1;
    // int tiles = (width + pad - kernel) / output_tile + 1;
    // return (tiles - 1) * output_tile + tile - width;
    int pad_width = (width + 2 * pad - kernel) / output_tile * output_tile;
    return pad_width + tile - width;
  };

  math::PadFunctor<CPU, float> pad;
  Tensor input_pad;
  framework::Tensor transformed_input;
  for (int i = 0; i < batch_size; ++i) {
    Tensor in_batch = input->Slice(i, i + 1);
    Tensor out_batch = output->Slice(i, i + 1);
    // int pad_bottom = winograd_pad(in_batch.dims()[2], paddings[0]);
    // int pad_right = winograd_pad(in_batch.dims()[3], paddings[1]);
    int pad_bottom = paddings[0];
    int pad_right = paddings[1];
    if (paddings[0] || paddings[1] || pad_bottom || pad_right) {
      framework::DDim pad_shape = in_batch.dims();
      pad_shape[2] += paddings[0] + pad_bottom;
      pad_shape[3] += paddings[1] + pad_right;
      input_pad.mutable_data<float>(pad_shape);
      pad(in_batch, paddings[0], pad_bottom, paddings[1], pad_right,
          &input_pad);
    } else {
      input_pad = in_batch;
    }
    // tile input and transform
    math::winograd_transform_input<tile, kernel>(input_pad, &transformed_input);
    // caculate output
    math::winograd_transform_output<tile, kernel>(transformed_input, *filter,
                                                  output);
  }
}

template <typename Itype, typename Otype>
void DepthwiseConv3x3(const ConvParam<CPU> &param) {
  const Tensor *input = param.Input();
  const Tensor *filter = param.Filter();
  const std::vector<int> &paddings = param.Paddings();
  const std::vector<int> &strides = param.Strides();
  const int batch_size = input->dims()[0];
  Tensor *output = param.Output();
  output->mutable_data<Otype>();

  if (strides[0] == 1) {
    for (int i = 0; i < batch_size; i++) {
      Tensor in_batch = input->Slice(i, i + 1);
      Tensor out_batch = output->Slice(i, i + 1);
      math::DepthwiseConv3x3S1<Itype, Otype>(in_batch, *filter, paddings,
                                             &out_batch);
    }
  } else if (strides[0] == 2) {
    for (int i = 0; i < batch_size; i++) {
      Tensor in_batch = input->Slice(i, i + 1);
      Tensor out_batch = output->Slice(i, i + 1);
      math::DepthwiseConv3x3S2<Itype, Otype>(in_batch, *filter, paddings,
                                             &out_batch);
    }
  } else {
    GemmConv<Itype, Otype>(param);
  }
}

template <typename Itype, typename Otype>
void DepthwiseConv5x5(const ConvParam<CPU> &param) {
  const Tensor *input = param.Input();
  const Tensor *filter = param.Filter();
  const std::vector<int> &paddings = param.Paddings();
  const std::vector<int> &strides = param.Strides();
  const int batch_size = input->dims()[0];
  Tensor *output = param.Output();
  output->mutable_data<Otype>();

  if (strides[0] == 1) {
    for (int i = 0; i < batch_size; i++) {
      Tensor in_batch = input->Slice(i, i + 1);
      Tensor out_batch = output->Slice(i, i + 1);
      math::DepthwiseConv5x5S1<Itype, Otype>(in_batch, *filter, paddings,
                                             &out_batch);
    }
  } else {
    GemmConv<Itype, Otype>(param);
  }
}

template void GemmConv<float, float>(const ConvParam<CPU> &param);
template void WinogradConv3x3<8, 3>(const ConvParam<CPU> &param);
template void DepthwiseConv3x3<float, float>(const ConvParam<CPU> &param);
template void DepthwiseConv5x5<float, float>(const ConvParam<CPU> &param);

#ifndef __aarch64__
template void GemmConv<int8_t, int32_t>(const ConvParam<CPU> &param);
template void DepthwiseConv3x3<int8_t, int32_t>(const ConvParam<CPU> &param);
template void DepthwiseConv5x5<int8_t, int32_t>(const ConvParam<CPU> &param);
#endif

}  // namespace operators
}  // namespace paddle_mobile
