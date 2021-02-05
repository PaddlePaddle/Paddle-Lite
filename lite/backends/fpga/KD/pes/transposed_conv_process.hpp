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
  Shape shape(NCHW,
              {cnhw_shape.channel(),
               cnhw_shape.num(),
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

void fill_sub_filters(ConvParam* param, Tensor* filter) {
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
    float16* out_address = nullptr;
    float* out_scale_address = nullptr;

    Shape shape_nchw(NCHW, {sub_num, channel, sub_h, sub_w});
    Shape shape_nhwc(NHWC, {sub_num, sub_h, sub_w, channel});

    BasicConvParam* basic_conv_param = new BasicConvParam();
    basic_conv_param->output.setDataLocation(Device);
    Shape tmp_shape(NHWC, {1, 1, 1, 1});
    basic_conv_param->output.mutableData<float16>(FP16, tmp_shape);

    basic_conv_param->filter.mutableData<int8_t>(INT8, shape_nchw);
    Tensor float_tensor;
    float* sub_filter_data = float_tensor.mutableData<float>(FP32, shape_nchw);

    for (int nn = 0; nn < sub_num; ++nn) {
      int ni = nn % kernel_num;
      int woff = sub_conv_number - 1 - (nn / kernel_num);
      for (int cc = 0; cc < channel; ++cc) {
        for (int hh = 0; hh < sub_h; ++hh) {
          int hi = hh * sub_conv_number + i;
          for (int ww = 0; ww < sub_w; ++ww) {
            int wi = ww * sub_conv_number + woff;
            int sidx = ((nn * channel * sub_h + cc * sub_h + hh) * sub_w + ww);
            int kidx =
                ((ni * channel * height + cc * height + hi) * width + wi);
            memcpy(sub_filter_data + sidx, filter_data + kidx, sizeof(float));
          }
        }
      }
    }

    float_tensor.flush();
    std::vector<float> quant_scale;
    format_filter(&float_tensor,
                  &(basic_conv_param->filter),
                  param->groups,
                  quant_scale,
                  max);

    Tensor scale;
    Tensor bias;
    Shape s_shape(NC, {1, sub_num});
    float* scale_data = scale.mutableData<float>(FP32, s_shape);
    float* bias_data = bias.mutableData<float>(FP32, s_shape);

    for (int n = 0; n < sub_num; n++) {
      int q_idx = (n + omit_size * kernel_num) % sub_num;
      scale_data[n] =
          param->scale()->data<float>()[n % kernel_num] * quant_scale[q_idx];
    }

    for (int n = 0; n < sub_num; n++) {
      bias_data[n] = param->bias()->data<float>()[n % kernel_num];
    }

    format_bias_scale_new(&bias, &scale, &basic_conv_param->scaleBias);
    basic_conv_param->scaleBias.flush();
    ConvArgs& args = basic_conv_param->args;
    int offset = (sub_conv_number - 1 - i) *
                 align_to_x(after_omit_out_w * kernel_num, 16);
    out_address = output->data<float16>() + offset;
    out_scale_address = basic_conv_param->output.scale();

    args.group_num = param->groups;
    args.sb_address = basic_conv_param->scaleBias.data<float16>();
    args.kernel.stride_h = 1;
    args.kernel.stride_w = 1;
    args.kernel.height = sub_h;
    args.kernel.width = sub_w;

    args.filter_address = basic_conv_param->filter.data<int8_t>();
    args.filter_num = sub_num;
    args.filter_scale_address = basic_conv_param->filter.scale();
    args.image.address = input->data<void>();
    args.image.scale_address = input->scale();
    args.image.channels = channel;
    args.image.width = input->shape().width();
    args.image.height = input->shape().height();
    args.image.pad_width = sub_pad;
    args.image.pad_height = sub_pad;
    args.dilation = param->dilations[0];
    args.output.address = out_address;
    args.output.scale_address = out_scale_address;
    args.deconv.enabled = 1;
    args.deconv.sub_kernel_num = sub_conv_number;
    args.deconv.invalid_col_num = omit_size;

    param->splitParams().push_back(basic_conv_param);
  }
}

}  // namespace zynqmp
}  // namespace paddle
