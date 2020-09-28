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
//
// Part of the following code in this file refs to
// https://github.com/msracver/Deformable-ConvNets/blob/master/faster_rcnn/operator_cxx/deformable_convolution.cu
//
// Copyright (c) 2017 Microsoft
// Licensed under The Apache-2.0 License [see LICENSE for details]
// \file deformable_psroi_pooling.cu
// \brief
// \author Yi Li, Guodong Zhang, Jifeng Dai

/**
 * @note: all code in this file are copied from paddle fluid
 * paddle commit id: f4c750d721a1226738bea382f6c0cf725cca8481
 *
 * check "paddle/fluid/operators/deformable_conv_op.h"
 * and "paddle/fluid/operators/deformable_conv_func.h"
 * if necessary
 */

#pragma once

#include <math.h>

#include <algorithm>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

template <typename T>
HOSTDEVICE T DmcnGetGradientWeight(T argmax_h,
                                   T argmax_w,
                                   const int h,
                                   const int w,
                                   const int height,
                                   const int width) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  weight = (h == argmax_h_low && w == argmax_w_low)
               ? (h + 1 - argmax_h) * (w + 1 - argmax_w)
               : weight;
  weight = (h == argmax_h_low && w == argmax_w_high)
               ? (h + 1 - argmax_h) * (argmax_w + 1 - w)
               : weight;
  weight = (h == argmax_h_high && w == argmax_w_low)
               ? (argmax_h + 1 - h) * (w + 1 - argmax_w)
               : weight;
  weight = (h == argmax_h_high && w == argmax_w_high)
               ? (argmax_h + 1 - h) * (argmax_w + 1 - w)
               : weight;

  return weight;
}

template <typename T>
HOSTDEVICE T DmcnGetCoordinateWeight(T argmax_h,
                                     T argmax_w,
                                     const int height,
                                     const int width,
                                     const T* im_data,
                                     const int data_width,
                                     const int bp_dir) {
  if (argmax_h <= -1 || argmax_h >= height || argmax_w <= -1 ||
      argmax_w >= width) {
    return 0;
  }

  int argmax_h_low = floor(argmax_h);
  int argmax_w_low = floor(argmax_w);
  int argmax_h_high = argmax_h_low + 1;
  int argmax_w_high = argmax_w_low + 1;

  T weight = 0;

  if (bp_dir == 0) {
    weight += (argmax_h_low >= 0 && argmax_w_low >= 0)
                  ? -1 * (argmax_w_low + 1 - argmax_w) *
                        im_data[argmax_h_low * data_width + argmax_w_low]
                  : 0;

    weight += (argmax_h_low >= 0 && argmax_w_high <= width - 1)
                  ? -1 * (argmax_w - argmax_w_low) *
                        im_data[argmax_h_low * data_width + argmax_w_high]
                  : 0;

    weight += (argmax_h_high <= height - 1 && argmax_w_low >= 0)
                  ? (argmax_w_low + 1 - argmax_w) *
                        im_data[argmax_h_high * data_width + argmax_w_low]
                  : 0;
    weight += (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
                  ? (argmax_w - argmax_w_low) *
                        im_data[argmax_h_high * data_width + argmax_w_high]
                  : 0;
  } else if (bp_dir == 1) {
    weight += (argmax_h_low >= 0 && argmax_w_low >= 0)
                  ? -1 * (argmax_h_low + 1 - argmax_h) *
                        im_data[argmax_h_low * data_width + argmax_w_low]
                  : 0;
    weight += (argmax_h_low >= 0 && argmax_w_high <= width - 1)
                  ? (argmax_h_low + 1 - argmax_h) *
                        im_data[argmax_h_low * data_width + argmax_w_high]
                  : 0;
    weight += (argmax_h_high <= height - 1 && argmax_w_low >= 0)
                  ? -1 * (argmax_h - argmax_h_low) *
                        im_data[argmax_h_high * data_width + argmax_w_low]
                  : 0;
    weight += (argmax_h_high <= height - 1 && argmax_w_high <= width - 1)
                  ? (argmax_h - argmax_h_low) *
                        im_data[argmax_h_high * data_width + argmax_w_high]
                  : 0;
  }

  return weight;
}

template <typename T>
HOSTDEVICE T DmcnIm2colBilinear(const T* bottom_data,
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
void ModulatedDeformableCol2imCPUKernel(const int num_kernels,
                                        const T* data_col,
                                        const T* data_offset,
                                        const T* data_mask,
                                        const int channels,
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
                                        const int deformable_group,
                                        const int height_col,
                                        const int width_col,
                                        T* grad_im) {
  for (int thread = 0; thread < num_kernels; thread++) {
    const int j = (thread / width_col / height_col / batch_size) % kernel_w;
    const int i =
        (thread / width_col / height_col / batch_size / kernel_w) % kernel_h;
    const int c =
        thread / width_col / height_col / batch_size / kernel_w / kernel_h;

    const int deformable_group_index = c / channel_per_deformable_group;

    int w_out = thread % width_col;
    int h_out = (thread / width_col) % height_col;
    int b = (thread / width_col / height_col) % batch_size;
    int w_in = w_out * stride_w - pad_w;
    int h_in = h_out * stride_h - pad_h;

    const T* data_offset_ptr = data_offset +
                               (b * deformable_group + deformable_group_index) *
                                   2 * kernel_h * kernel_w * height_col *
                                   width_col;
    const T* data_mask_ptr = data_mask +
                             (b * deformable_group + deformable_group_index) *
                                 kernel_h * kernel_w * height_col * width_col;
    const int data_offset_h_ptr =
        ((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out;
    const int data_offset_w_ptr =
        ((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col + w_out;
    const int data_mask_hw_ptr =
        ((i * kernel_w + j) * height_col + h_out) * width_col + w_out;
    const T offset_h = data_offset_ptr[data_offset_h_ptr];
    const T offset_w = data_offset_ptr[data_offset_w_ptr];
    const T mask = data_mask_ptr[data_mask_hw_ptr];
    const T cur_inv_h_data = h_in + i * dilation_h + offset_h;
    const T cur_inv_w_data = w_in + j * dilation_w + offset_w;

    const T cur_top_grad = data_col[thread] * mask;
    const int cur_h = static_cast<int>(cur_inv_h_data);
    const int cur_w = static_cast<int>(cur_inv_w_data);
    for (int dy = -2; dy <= 2; dy++) {
      for (int dx = -2; dx <= 2; dx++) {
        if (cur_h + dy >= 0 && cur_h + dy < height && cur_w + dx >= 0 &&
            cur_w + dx < width && abs(cur_inv_h_data - (cur_h + dy)) < 1 &&
            abs(cur_inv_w_data - (cur_w + dx)) < 1) {
          int cur_bottom_grad_pos =
              ((b * channels + c) * height + cur_h + dy) * width + cur_w + dx;
          T weight = DmcnGetGradientWeight(cur_inv_h_data,
                                           cur_inv_w_data,
                                           cur_h + dy,
                                           cur_w + dx,
                                           height,
                                           width);

          *(grad_im + cur_bottom_grad_pos) =
              *(grad_im + cur_bottom_grad_pos) + weight * cur_top_grad;
        }
      }
    }
  }
}

template <typename T>
static inline void ModulatedDeformableCol2imCPU(
    const T* data_col,
    const T* data_offset,
    const T* data_mask,
    const std::vector<int64_t> im_shape,
    const std::vector<int64_t> col_shape,
    const std::vector<int64_t> kernel_shape,
    const std::vector<int> pad,
    const std::vector<int> stride,
    const std::vector<int> dilation,
    const int deformable_group,
    T* grad_im) {
  int channel_per_deformable_group = im_shape[0] / deformable_group;
  int num_kernels = col_shape[0] * col_shape[1] * col_shape[2] * col_shape[3];

  ModulatedDeformableCol2imCPUKernel(num_kernels,
                                     data_col,
                                     data_offset,
                                     data_mask,
                                     im_shape[0],
                                     im_shape[1],
                                     im_shape[2],
                                     kernel_shape[2],
                                     kernel_shape[3],
                                     pad[0],
                                     pad[1],
                                     stride[0],
                                     stride[1],
                                     dilation[0],
                                     dilation[1],
                                     channel_per_deformable_group,
                                     col_shape[1],
                                     deformable_group,
                                     col_shape[2],
                                     col_shape[3],
                                     grad_im);
}

template <typename T>
void ModulatedDeformableCol2imCoordCPUKernel(
    const int num_kernels,
    const T* data_col,
    const T* data_im,
    const T* data_offset,
    const T* data_mask,
    const int channels,
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
    const int offset_channels,
    const int deformable_group,
    const int height_col,
    const int width_col,
    T* grad_offset,
    T* grad_mask) {
  for (int i = 0; i < num_kernels; i++) {
    T val = 0, mval = 0;
    const int w = i % width_col;
    const int h = (i / width_col) % height_col;
    const int c = (i / width_col / height_col) % offset_channels;
    const int b = (i / width_col / height_col) / offset_channels;

    const int deformable_group_index = c / (2 * kernel_h * kernel_w);
    const int col_step = kernel_h * kernel_w;
    int cnt = 0;
    const T* data_col_ptr = data_col +
                            deformable_group_index *
                                channel_per_deformable_group * batch_size *
                                width_col * height_col;
    const T* data_im_ptr = data_im +
                           (b * deformable_group + deformable_group_index) *
                               channel_per_deformable_group / kernel_h /
                               kernel_w * height * width;
    const T* data_offset_ptr = data_offset +
                               (b * deformable_group + deformable_group_index) *
                                   2 * kernel_h * kernel_w * height_col *
                                   width_col;
    const T* data_mask_ptr = data_mask +
                             (b * deformable_group + deformable_group_index) *
                                 kernel_h * kernel_w * height_col * width_col;

    const int offset_c = c - deformable_group_index * 2 * kernel_h * kernel_w;

    for (int col_c = offset_c / 2; col_c < channel_per_deformable_group;
         col_c += col_step) {
      const int col_pos =
          (((col_c * batch_size + b) * height_col) + h) * width_col + w;
      const int bp_dir = offset_c % 2;

      int j = (col_pos / width_col / height_col / batch_size) % kernel_w;
      int i =
          (col_pos / width_col / height_col / batch_size / kernel_w) % kernel_h;
      int w_out = col_pos % width_col;
      int h_out = (col_pos / width_col) % height_col;
      int w_in = w_out * stride_w - pad_w;
      int h_in = h_out * stride_h - pad_h;
      const int data_offset_h_ptr =
          (((2 * (i * kernel_w + j)) * height_col + h_out) * width_col + w_out);
      const int data_offset_w_ptr =
          (((2 * (i * kernel_w + j) + 1) * height_col + h_out) * width_col +
           w_out);
      const int data_mask_hw_ptr =
          (((i * kernel_w + j) * height_col + h_out) * width_col + w_out);
      const T offset_h = data_offset_ptr[data_offset_h_ptr];
      const T offset_w = data_offset_ptr[data_offset_w_ptr];
      const T mask = data_mask_ptr[data_mask_hw_ptr];
      T inv_h = h_in + i * dilation_h + offset_h;
      T inv_w = w_in + j * dilation_w + offset_w;
      if (inv_h <= -1 || inv_w <= -1 || inv_h >= height || inv_w >= width) {
        inv_h = inv_w = -2;
      } else {
        mval += data_col_ptr[col_pos] *
                DmcnIm2colBilinear(data_im_ptr + cnt * height * width,
                                   width,
                                   height,
                                   width,
                                   inv_h,
                                   inv_w);
      }
      const T weight =
          DmcnGetCoordinateWeight(inv_h,
                                  inv_w,
                                  height,
                                  width,
                                  data_im_ptr + cnt * height * width,
                                  width,
                                  bp_dir);
      val += weight * data_col_ptr[col_pos] * mask;
      cnt += 1;
    }
    grad_offset[i] = val;
    if (offset_c % 2 == 0)
      grad_mask[(((b * deformable_group + deformable_group_index) * kernel_h *
                      kernel_w +
                  offset_c / 2) *
                     height_col +
                 h) *
                    width_col +
                w] = mval;
  }
}

template <typename T>
static inline void ModulatedDeformableCol2imCoordCPU(
    const T* data_col,
    const T* data_im,
    const T* data_offset,
    const T* data_mask,
    const std::vector<int64_t> im_shape,
    const std::vector<int64_t> col_shape,
    const std::vector<int64_t> kernel_shape,
    const std::vector<int> paddings,
    const std::vector<int> strides,
    const std::vector<int> dilations,
    const int deformable_groups,
    T* grad_offset,
    T* grad_mask) {
  int num_kernels = 2 * kernel_shape[2] * kernel_shape[3] * col_shape[1] *
                    col_shape[2] * col_shape[3] * deformable_groups;
  int channel_per_deformable_group = col_shape[0] / deformable_groups;

  ModulatedDeformableCol2imCoordCPUKernel(
      num_kernels,
      data_col,
      data_im,
      data_offset,
      data_mask,
      im_shape[0],
      im_shape[1],
      im_shape[2],
      kernel_shape[2],
      kernel_shape[3],
      paddings[0],
      paddings[1],
      strides[0],
      strides[1],
      dilations[0],
      dilations[1],
      channel_per_deformable_group,
      col_shape[1],
      2 * kernel_shape[2] * kernel_shape[3] * deformable_groups,
      deformable_groups,
      col_shape[2],
      col_shape[3],
      grad_offset,
      grad_mask);
}

template <typename T>
void ModulatedDeformableIm2colCPUKernel(const int num_kernels,
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

template <typename T>
void FilterGradAddupCPUKernel(const int nthreads,
                              const int n,
                              const int height,
                              const int width,
                              const T* dweight_3d,
                              T* filter_grad) {
  for (int i = 0; i < nthreads; i++) {
    filter_grad[i] = filter_grad[i] + dweight_3d[i];
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
