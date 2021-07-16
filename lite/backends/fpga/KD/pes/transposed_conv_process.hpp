/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <string.h>
#include <cmath>
#include <vector>

#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/backends/fpga/KD/pes/conv_process.hpp"
#include "lite/backends/fpga/KD/tensor.hpp"
#include "lite/backends/fpga/KD/util.hpp"

namespace paddle {
namespace zynqmp {
/*
calculate sub padding number
*/
inline int calc_sub_pad(int filter_axis, int pad, int stride) {
  if (stride == 0 || ((filter_axis - pad - 1) < 0)) {
    ENFORCE(false, "Wrong deconv parameters");
  }
  return (filter_axis - pad - 1) / stride;
}

inline int get_sub_filter_axis(int filter_axis, int stride) {
  return (filter_axis / stride);
}

inline int get_sub_out_axis(int image_axis, int sub_pad, int sub_filter_axis) {
  return ((image_axis + 2 * sub_pad - sub_filter_axis) + 1);
}

/*
(filter_width-pad,filter_width-pad) is the first pixel of sub-pixel image
position. so the omit rows or columns is (stride - )
*/
inline int deconv_get_omit(int stride, int filter_width, int pad) {
  ENFORCE(filter_width > pad, "invalid deconv parameters");
  int idx;
  bool flag = false;
  for (idx = 1; idx <= stride; ++idx) {
    int j = idx;
    for (; j <= filter_width;) {
      if (j == filter_width - pad) {
        flag = true;
        break;
      }
      j = j + stride;
    }
    if (flag) {
      break;
    }
  }
  return (stride - idx);
}
/**
filter data in PaddlePaddle is CNHW format.
this function convert it into NCHW format.
*/
void inline convert_cnhw_to_nchw(Tensor* cnhw, Tensor* nchw) {
  Shape& cnhw_shape = cnhw->shape();
  // For all param tensors loaded from the param file, the shapes are all
  // treated as N, C, H, W
  // So here cnhw_shape.channel() is actually filter num.
  // Is this a good way?
  Shape shape(NCHW,
              {cnhw_shape.num(),
               cnhw_shape.channel(),
               cnhw_shape.height(),
               cnhw_shape.width()});
  float* nchw_data = nchw->mutableData<float>(FP32, shape);
  float* cnhw_data = cnhw->data<float>();

  int hw = shape.height() * shape.width();
  int nhw = shape.num() * hw;
  int chw = shape.channel() * hw;

  int index = 0;
  for (int c = 0; c < shape.channel(); c++) {
    for (int n = 0; n < shape.num(); n++) {
      float* dst = nchw_data + c * hw + n * chw;
      float* src = cnhw_data + index;
      memcpy(dst, src, hw * sizeof(float));
      index += hw;
    }
  }
}

template <typename T>
void inline nchw_to_nhwc(Tensor* nchw, Tensor* nhwc) {
  Shape& shape = nchw->shape();
  T* x = nchw->data<T>();
  T* y = nhwc->data<T>();

  int num = shape.num();
  int channel = shape.channel();
  int height = shape.height();
  int width = shape.width();
  int index = 0;
  int hwc = height * width * channel;
  int wc = width * channel;
  for (int n = 0; n < num; n++) {
    int base = n * hwc;
    for (int c = 0; c < channel; c++) {
      for (int h = 0; h < height; h++) {
        int offset_height = h * wc;
        for (int w = 0; w < width; w++) {
          int dst_index = base + offset_height + w * channel + c;
          y[dst_index] = x[index];
          index++;
        }
      }
    }
  }
}

template <typename T>
void inline nhwc_to_nchw(Tensor* nhwc, Tensor* nchw) {
  Shape& shape = nhwc->shape();
  T* x = nhwc->data<T>();
  T* y = nchw->data<T>();

  int num = shape.num();
  int channel = shape.channel();
  int height = shape.height();
  int width = shape.width();
  int index = 0;
  int hwc = height * width * channel;
  int hw = height * width;
  for (int n = 0; n < num; n++) {
    int base = n * hwc;
    for (int h = 0; h < height; h++) {
      for (int w = 0; w < width; w++) {
        for (int c = 0; c < channel; c++) {
          int dst_index = base + c * hw + h * width + w;
          y[dst_index] = x[index];
          index++;
        }
      }
    }
  }
}

template <typename T>
void inline hwc_to_chw(Tensor* hwc, Tensor* chw) {
  Shape& shape = chw->shape();
  T* x = hwc->data<T>();
  T* y = chw->data<T>();

  int channel = shape.channel();
  int height = shape.height();
  int width = shape.width();
  int index = 0;
  int wc = width * channel;
  int hw = height * width;
  for (int h = 0; h < height; h++) {
    for (int w = 0; w < width; w++) {
      for (int c = 0; c < channel; c++) {
        int dst_index = c * hw + h * width + w;
        y[dst_index] = x[index];
        index++;
      }
    }
  }
}

/**
inverse data in one HW plane,take 3x3 filter for example
0<->8, 1<-->7, 2<-->6, 3<-->5
*/
void inline inverse_filter(Tensor* tensor) {
  float* data = tensor->data<float>();
  Shape& shape = tensor->shape();
  int hw = shape.height() * shape.width();
  int chw = shape.channel() * hw;

  int hw_last_index = hw - 1;
  for (int n = 0; n < shape.num(); n++) {
    for (int c = 0; c < shape.channel(); c++) {
      float* hw_start = data + n * chw + c * hw;
      float tmp_data;
      for (int i = 0; i < hw / 2; i++) {
        tmp_data = hw_start[i];
        hw_start[i] = hw_start[hw_last_index - i];
        hw_start[hw_last_index - i] = tmp_data;
      }
    }
  }
}

DCpuConcatType fill_sub_filters(ConvParam* param, Tensor* filter) {
  DCpuConcatType deconv_concat_type = DCpuConcatType::DISABLED;
  int dynamic_range = 127;  // int8 max value
  float16 dynamic_range_fp16 = float_to_half(dynamic_range * 1.0);
  float inv_dynamic_range = 1.0 / dynamic_range;

  Tensor* input = param->input;
  Tensor* output = param->output;
  int sub_conv_number = param->strides[0];
  int kernel_num = filter->shape().num();
  int height = filter->shape().height();
  int width = filter->shape().width();
  int sub_num = kernel_num * sub_conv_number;
  int sub_h = height / sub_conv_number;
  int sub_w = width / sub_conv_number;
  int sub_pad = calc_sub_pad(width, param->paddings[0], param->strides[0]);
  int omit_size = deconv_get_omit(param->strides[0], width, param->paddings[0]);
  int channel = filter->shape().channel();
  int sub_output_w = get_sub_out_axis(input->shape().width(), sub_pad, sub_w);
  int sub_output_h = get_sub_out_axis(input->shape().height(), sub_pad, sub_h);
  int before_omit_out_w = sub_output_w * sub_conv_number;
  int before_omit_out_h = sub_output_h * sub_conv_number;
  int after_omit_out_w = before_omit_out_w - 2 * omit_size;
  int after_omit_out_h = before_omit_out_h - 2 * omit_size;

  float max = find_max(*filter);
  float mem_factor = before_omit_out_h * 1.0 / after_omit_out_h;
  output->setMemScale(mem_factor);
  output->mutableData<float16>();
  float* filter_data = filter->data<float>();

  for (int i = 0; i < sub_conv_number; i++) {
    //  init a ConvParam for a sub filter conv
    ConvParam sub_param;
    sub_param.input = input;
    sub_param.output = output;
    sub_param.groups = param->groups;
    sub_param.strides = std::vector<int>({1, 1});
    sub_param.paddings = std::vector<int>({sub_pad, sub_pad});
    sub_param.dilations = param->dilations;

    // TODO(chengruichang) There is an assumption that channel < 2047
    const Shape sub_filter_shape(NCHW, {sub_num, channel, sub_h, sub_w});
    // copy filter data into sub filters separately
    Tensor sub_filter;
    sub_param.filter = &sub_filter;
    float* sub_filter_data =
        sub_filter.mutableData<float>(FP32, sub_filter_shape);
    int idx_in_stride_h = i % sub_conv_number;
    float* dst = sub_filter_data;
    float* src = filter_data;
    for (int n = 0; n < sub_num; ++n) {
      for (int c = 0; c < channel; ++c) {
        for (int h = 0; h < sub_h; ++h) {
          for (int w = 0; w < sub_w; ++w) {
            // Sub filters along h axis are arranged in reversed order
            int idx_in_stride_w = sub_conv_number - 1 - n / kernel_num;
            int original_n = n % kernel_num;
            int original_h = h * sub_conv_number + idx_in_stride_h;
            int original_w = w * sub_conv_number + idx_in_stride_w;

            dst = sub_filter_data + w + h * sub_w + c * sub_w * sub_h +
                  n * sub_w * sub_h * channel;
            src = filter_data + original_w + original_h * width +
                  c * width * height + original_n * width * height * channel;
            memcpy(dst, src, sizeof(float));
          }
        }
      }
    }
    sub_param.filter->flush();

    Tensor* sub_scale = sub_param.scale();
    Tensor* sub_bias = sub_param.bias();
    Shape s_shape(NC, {1, sub_num});
    float* scale_data = sub_scale->mutableData<float>(FP32, s_shape);
    float* bias_data = sub_bias->mutableData<float>(FP32, s_shape);
    for (int n = 0; n < sub_num; n++) {
      scale_data[n] = param->scale()->data<float>()[n % kernel_num];
    }
    for (int n = 0; n < sub_num; n++) {
      bias_data[n] = param->bias()->data<float>()[n % kernel_num];
    }
    sub_scale->flush();
    sub_bias->flush();

    int start_offset = (sub_conv_number - 1 - i) *
                       align_to_x(after_omit_out_w * kernel_num, 16);
    const ConvParam& sb_param = sub_param;

    // Filter data loaded from params is NCHW.
    // Format transform is made in split_filter_num.
    deconv_concat_type = split_filter_num(
        sb_param, start_offset, true, kernel_num, omit_size, sub_conv_number);

    for (auto basic_conv_param :
         const_cast<ConvParam&>(sb_param).splitParams()) {
      param->splitParams().push_back(basic_conv_param);
    }
    const_cast<ConvParam&>(sb_param).splitParams().clear();
  }
  return deconv_concat_type;
}

}  // namespace zynqmp
}  // namespace paddle
