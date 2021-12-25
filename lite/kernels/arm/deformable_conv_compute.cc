// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/arm/deformable_conv_compute.h"
#include <cmath>
#include <utility>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/kernels/arm/conv_depthwise.h"
#include "lite/kernels/arm/conv_direct.h"
#include "lite/kernels/arm/conv_gemmlike.h"
#include "lite/kernels/arm/conv_winograd.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <>
void DeformableConvCompute<PRECISION(kFloat),
                           PRECISION(kFloat)>::PrepareForRun() {
  ReInitWhenNeeded();
}

template <typename T>
static T DmcnIm2colBilinear(const T* bottom_data,
                            const int data_width,
                            const int height,
                            const int width,
                            T h,
                            T w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh;
  T hw = 1 - lw;

  T v1 =
      (h_low >= 0 && w_low >= 0) ? bottom_data[h_low * data_width + w_low] : 0;
  T v2 = (h_low >= 0 && w_high <= width - 1)
             ? bottom_data[h_low * data_width + w_high]
             : 0;
  T v3 = (h_high <= height - 1 && w_low >= 0)
             ? bottom_data[h_high * data_width + w_low]
             : 0;
  T v4 = (h_high <= height - 1 && w_high <= width - 1)
             ? bottom_data[h_high * data_width + w_high]
             : 0;

  T w1 = hh * hw;
  T w2 = hh * lw;
  T w3 = lh * hw;
  T w4 = lh * lw;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename T>
static void ModulatedDeformableIm2colCPUKernel(
    const int num_kernels,
    const T* data_im,
    const T* data_offset,
    const T* data_mask,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size,
    const int num_channels,
    const int deformable_group,
    const int height_col,
    const int width_col,
    T* data_col) {
  for (int i = 0; i < num_kernels; i++) {
    const int w_col = i % width_col;
    const int h_col = (i / width_col) % height_col;
    const int b_col = (i / width_col) / height_col % batch_size;
    const int c_im = (i / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    T* data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const T* data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const T* data_offset_ptr =
        data_offset +
        (b_col * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;
    const T* data_mask_ptr =
        data_mask +
        (b_col * deformable_group + deformable_group_index) * kernel_h *
            kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const int data_mask_hw_ptr =
            ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;

        const T offset_h = data_offset_ptr[data_offset_h_ptr];
        const T offset_w = data_offset_ptr[data_offset_w_ptr];
        const T mask = data_mask_ptr[data_mask_hw_ptr];
        T val = static_cast<T>(0);
        const T h_im = h_in + i * dilation_h + offset_h;
        const T w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          val =
              DmcnIm2colBilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename T>
static inline void ModulatedDeformableIm2colCPU(
    const T* data_im,
    const T* data_offset,
    const T* data_mask,
    const std::vector<int64_t> im_shape,
    const std::vector<int64_t> col_shape,
    const std::vector<int64_t> filter_shape,
    const std::vector<int> paddings,
    const std::vector<int> strides,
    const std::vector<int> dilations,
    const int deformable_groups,
    T* data_col) {
  int channel_per_deformable_group = im_shape[0] / deformable_groups;
  int num_kernels = im_shape[0] * col_shape[1] * col_shape[2] * col_shape[3];

  // get outputs of im2col with offset by bilinear interpolation
  ModulatedDeformableIm2colCPUKernel(num_kernels,
                                     data_im,
                                     data_offset,
                                     data_mask,
                                     im_shape[1],
                                     im_shape[2],
                                     filter_shape[2],
                                     filter_shape[3],
                                     paddings[0],
                                     paddings[1],
                                     strides[0],
                                     strides[1],
                                     dilations[0],
                                     dilations[1],
                                     channel_per_deformable_group,
                                     col_shape[1],
                                     im_shape[0],
                                     deformable_groups,
                                     col_shape[2],
                                     col_shape[3],
                                     data_col);
}

template <>
void DeformableConvCompute<PRECISION(kFloat), PRECISION(kFloat)>::Run() {
  const auto& param = this->Param<operators::DeformableConvParam>();
  auto& ctx = this->ctx_->template As<ARMContext>();
  const auto* input = param.x;
  const auto* offset = param.offset;
  const auto* mask = param.mask;
  const auto& filter = *param.conv_param.filter;
  const auto* filter_data = param.conv_param.filter->data<float>();
  if (flag_trans_weights_) {
    filter_data = weights_.data<float>();
  }
  auto* output = param.output;
  bool is_bias = param.conv_param.bias ? true : false;
  const float* bias =
      param.conv_param.bias ? param.conv_param.bias->data<float>() : nullptr;

  const int groups = param.conv_param.groups;
  const int deformable_groups = param.deformable_groups;
  const int im2col_step = param.im2col_step;
  const std::vector<int>& strides = param.conv_param.strides;
  const std::vector<int>& paddings = *param.conv_param.paddings;
  const std::vector<int>& dilations = *param.conv_param.dilations;

  const int batch_size = static_cast<int>(input->dims()[0]);

  std::vector<int64_t> filter_shape_vec(filter.dims().Vectorize());
  std::vector<int64_t> output_shape_vec(output->dims().Vectorize());

  // col_shape_vec: {c_i * k_h * k_w, im2col_step, o_h, o_w}
  std::vector<int64_t> col_buffer_shape_vec(filter_shape_vec.size());
  col_buffer_shape_vec[0] =
      input->dims()[1] * filter.dims()[2] * filter.dims()[3];
  col_buffer_shape_vec[1] = im2col_step;
  for (size_t j = 0; j < filter_shape_vec.size() - 2; ++j) {
    col_buffer_shape_vec[j + 2] = output_shape_vec[j + 2];
  }
  DDim col_shape(col_buffer_shape_vec);
  std::vector<int64_t> output_buffer_shape_vec(1);
  output_buffer_shape_vec[0] = batch_size * output_shape_vec[1] *
                               output_shape_vec[2] * output_shape_vec[3];
  DDim output_shape(output_buffer_shape_vec);
  Tensor col_buffer;
  Tensor output_buffer;
  col_buffer.Resize(col_shape);
  col_buffer.mutable_data<float>();
  output_buffer.Resize(output_shape);
  output_buffer.mutable_data<float>();
  int64_t M = output_shape_vec[1] / groups;
  int64_t N = im2col_step * output_shape_vec[2] * output_shape_vec[3];
  int64_t K =
      input->dims()[1] * filter_shape_vec[2] * filter_shape_vec[3] / groups;

  Tensor weight_3d;
  weight_3d.ShareDataWith(filter);
  weight_3d.Resize(DDim({groups, M, K}));
  Tensor col_buffer_3d;
  col_buffer_3d.ShareDataWith(col_buffer);
  col_buffer_3d.Resize(DDim({groups, K, N}));
  Tensor output_4d;
  output_4d.ShareDataWith(output_buffer);
  output_4d.Resize(DDim({batch_size / im2col_step, groups, M, N}));
  output_4d.mutable_data<float>();
  DDim input_shape = input->dims().Slice(1, input->dims().size());
  std::vector<int64_t> input_shape_vec = input_shape.Vectorize();
  int input_dim = input->numel() / input->dims()[0];
  int input_offset_dim = offset->numel() / offset->dims()[0];
  int input_mask_dim = mask->numel() / mask->dims()[0];
  const float* input_ptr = input->data<float>();
  const float* offset_ptr = offset->data<float>();
  const float* mask_ptr = mask->data<float>();
  col_buffer.mutable_data<float>();
  float* col_buffer_ptr = col_buffer.mutable_data<float>();
  int weights_size_per_group = M * K;
  for (int i = 0; i < batch_size / im2col_step; ++i) {
    ModulatedDeformableIm2colCPU<float>(
        input_ptr + i * im2col_step * input_dim,
        offset_ptr + i * im2col_step * input_offset_dim,
        mask_ptr + i * im2col_step * input_mask_dim,
        input_shape_vec,
        col_buffer_shape_vec,
        filter_shape_vec,
        paddings,
        strides,
        dilations,
        deformable_groups,
        col_buffer_ptr);
    Tensor output_3d = output_4d.Slice<float>(i, i + 1);
    output_3d.Resize(DDim(output_4d.dims()).Slice(1, output_4d.dims().size()));
    // get the product of pixel and weight
    for (int g = 0; g < groups; ++g) {
      const float* weights_group = filter_data + g * weights_size_per_group;
      const float* bias_group = bias + g * M;
      Tensor weight_3d_slice = weight_3d.Slice<float>(g, g + 1);
      weight_3d_slice.Resize(
          DDim(weight_3d.dims()).Slice(1, weight_3d.dims().size()));
      Tensor col_buffer_3d_slice = col_buffer_3d.Slice<float>(g, g + 1);
      col_buffer_3d_slice.Resize(
          DDim(col_buffer_3d.dims()).Slice(1, col_buffer_3d.dims().size()));
      Tensor output_3d_slice = output_3d.Slice<float>(g, g + 1);
      output_3d_slice.Resize(
          DDim(output_3d.dims()).Slice(1, output_3d.dims().size()));
      if (N == 1) {
        lite::arm::math::sgemv(
            weights_group,
            col_buffer_3d_slice.data<float>(),
            const_cast<float*>(output_3d_slice.data<float>()),
            false,
            M,
            K,
            0.f,
            is_bias,
            bias_group,
            param.conv_param.activation_param,
            &ctx);
      } else {
        lite::arm::math::sgemm_prepack(
            false,
            output_3d_slice.dims()[0],
            output_3d_slice.dims()[1],
            weight_3d_slice.dims()[1],
            weights_group,
            col_buffer_3d_slice.data<float>(),
            output_3d_slice.dims()[1],
            0.f,
            const_cast<float*>(output_3d_slice.data<float>()),
            output_3d_slice.dims()[1],
            bias_group,
            is_bias,
            param.conv_param.activation_param,
            &ctx);
      }
    }
  }
  output->ShareDataWith(output_buffer);
  output->Resize(DDim(output_shape_vec));
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

typedef paddle::lite::kernels::arm::DeformableConvCompute<PRECISION(kFloat),
                                                          PRECISION(kFloat)>
    DeformableConvFp32;

REGISTER_LITE_KERNEL(
    deformable_conv, kARM, kFloat, kNCHW, DeformableConvFp32, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Filter", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Offset", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
