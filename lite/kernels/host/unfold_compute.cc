// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/host/unfold_compute.h"
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

/**
 * The most common im2col algorithm.
 * Support dilation, stride and padding.
 *
 * im = [input_channels, input_height, input_width]
 * col = [input_channels, filter_height, filter_width, output_height,
 * output_width]
 */
template <typename T>
inline void im2col_common(const lite::Tensor& im,
                          const std::vector<int>& dilation,
                          const std::vector<int>& stride,
                          const std::vector<int>& padding,
                          lite::Tensor* col) {
  int im_channels = im.dims()[0];
  int im_height = im.dims()[1];
  int im_width = im.dims()[2];
  int filter_height = col->dims()[1];
  int filter_width = col->dims()[2];
  int output_height = col->dims()[3];
  int output_width = col->dims()[4];
  int channels_col = im_channels * filter_height * filter_width;

  const T* im_data = im.data<T>();
  T* col_data = col->template mutable_data<T>();
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % filter_width;
    int h_offset = (c / filter_width) % filter_height;
    int c_im = c / (filter_width * filter_height);
    for (int h = 0; h < output_height; ++h) {
      int im_row_idx = h * stride[0] - padding[0] + h_offset * dilation[0];
      for (int w = 0; w < output_width; ++w) {
        int im_col_idx = w * stride[1] - padding[1] + w_offset * dilation[1];
        int col_idx = (c * output_height + h) * output_width + w;
        int im_idx = (im_row_idx + c_im * im_height) * im_width + im_col_idx;
        col_data[col_idx] = (im_row_idx < 0 || im_row_idx >= im_height ||
                             im_col_idx < 0 || im_col_idx >= im_width)
                                ? static_cast<T>(0)
                                : im_data[im_idx];
      }
    }
  }
}

inline int CalcOutputSize(int input_size,
                          int filter_size,
                          int dilation,
                          int padding1,
                          int padding2,
                          int stride) {
  const int dkernel = dilation * (filter_size - 1) + 1;
  int output_size = (input_size + padding1 + padding2 - dkernel) / stride + 1;

  return output_size;
}

template <typename T, PrecisionType PType>
void UnfoldCompute<T, PType>::Run() {
  auto& param = this->template Param<operators::UnfoldParam>();
  const lite::Tensor* input = param.X;
  lite::Tensor* output = param.Y;
  auto input_dims = input->dims();
  const int batch_size = static_cast<int>(input_dims[0]);
  output->template mutable_data<T>();

  std::vector<int> kernel_sizes = param.kernel_sizes;
  std::vector<int> strides = param.strides;
  std::vector<int> paddings = param.paddings;
  std::vector<int> dilations = param.dilations;
  int output_height = CalcOutputSize(input_dims[2],
                                     kernel_sizes[0],
                                     dilations[0],
                                     paddings[0],
                                     paddings[2],
                                     strides[0]);
  int output_width = CalcOutputSize(input_dims[3],
                                    kernel_sizes[1],
                                    dilations[1],
                                    paddings[1],
                                    paddings[3],
                                    strides[1]);

  DDim input_shape({input_dims[1], input_dims[2], input_dims[3]});
  DDim output_matrix_shape({input_dims[1],
                            kernel_sizes[0],
                            kernel_sizes[1],
                            output_height,
                            output_width});
  for (int i = 0; i < batch_size; i++) {
    Tensor in_batch = input->template Slice<T>(i, i + 1);
    in_batch.Resize(input_shape);
    Tensor out_batch = output->template Slice<T>(i, i + 1);
    out_batch.Resize(output_matrix_shape);
    im2col_common<T>(in_batch, dilations, strides, paddings, &out_batch);
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

using unfold_float =
    paddle::lite::kernels::host::UnfoldCompute<float, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(unfold, kHost, kFloat, kNCHW, unfold_float, def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kFloat))})
    .Finalize();

using unfold_int32 =
    paddle::lite::kernels::host::UnfoldCompute<int, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(unfold, kHost, kFloat, kNCHW, unfold_int32, def_int32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();

using unfold_int64 =
    paddle::lite::kernels::host::UnfoldCompute<int64_t, PRECISION(kFloat)>;
REGISTER_LITE_KERNEL(unfold, kHost, kFloat, kNCHW, unfold_int64, def_int64)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .Finalize();

using unfold_int8 =
    paddle::lite::kernels::host::UnfoldCompute<int8_t, PRECISION(kInt8)>;
REGISTER_LITE_KERNEL(unfold, kHost, kInt8, kNCHW, unfold_int8, def_int8)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .Finalize();
