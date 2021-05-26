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

#include <algorithm>

#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/conv_process.hpp"

namespace paddle {
namespace zynqmp {

class DepthwiseConvPE : public PE {
 public:
  inline int gcd_(int a, int b) {
    while (b) {
      int temp = a;
      a = b;
      b = temp % b;
    }
    return a;
  }

  inline int lcm_(int a, int b) { return a * b / gcd_(a, b); }

  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    DepthwiseConvParam& param = param_;
    Tensor* input = param.input;
    Tensor* output = param.output;
    int channel = output->shape().channel();

    int image_dynamic_range = (1 << 11) - 1;  // int12 max value, pow(2,11)-1
    float16 dynamic_range_fp16 = float_to_half(image_dynamic_range * 1.0);
    float inv_dynamic_range = 1.0 / image_dynamic_range;

    int alignment = 16;

    if (channel % alignment != 0 || channel < alignment) {
      int c_lcm = lcm_(channel, alignment);
      align_repeat_ = c_lcm / (channel);
    }
    Shape shape(N, {2 * channel * align_repeat_});

    float16* b_data = scale_bias_.mutableData<float16>(FP16, shape);
    memset(b_data, 0, scale_bias_.memorySize());

    if (param_.bias()->dataType() == FP32) {
      float* new_bias_data = param_.bias()->data<float>();
      for (int i = 0; i < align_repeat_; i++) {
        for (int j = 0; j < channel; j++) {
          float16 value = float_to_half(new_bias_data[j]);
          b_data[i * channel + j] = value;
        }
      }
    } else {
      float16* new_bias_data = param_.bias()->data<float16>();
      for (int i = 0; i < align_repeat_; i++) {
        for (int j = 0; j < channel; j++) {
          b_data[i * channel + j] = new_bias_data[j];
        }
      }
    }

    if (param_.scale() == nullptr) {
      float16 one = float_to_half(1.0f);
      for (int i = 0; i < align_repeat_; i++) {
        for (int j = 0; j < channel; j++) {
          b_data[channel * align_repeat_ + i * channel + j] = one;
        }
      }
    } else {
      if (param_.scale()->dataType() == FP32) {
        float* new_scale_data = param_.scale()->data<float>();
        for (int i = 0; i < align_repeat_; i++) {
          for (int j = 0; j < channel; j++) {
            float16 value = float_to_half(new_scale_data[j]);
            b_data[channel * align_repeat_ + i * channel + j] = value;
          }
        }
      } else {
        float16* new_scale_data = param_.scale()->data<float16>();
        for (int i = 0; i < align_repeat_; i++) {
          for (int j = 0; j < channel; j++) {
            b_data[channel * align_repeat_ + i * channel + j] =
                new_scale_data[j];
          }
        }
      }
    }
    scale_bias_.flush();

    int filter_dynamic_range = 0;
    Tensor* null_scale = nullptr;

    if (param_.filter->shape().width() == 1 &&
        param_.filter->shape().height() == 1) {
      filter_dynamic_range = (1 << 15) - 1;  // int16 max value, pow(2,15)-1
    } else {
      int fix16_range = (1 << 15) - 1;
      int ext_range = (1 << 19) - 1;
      int max_area =
          static_cast<int>(ext_range / (param_.filter->shape().height() *
                                        param_.filter->shape().width()));
      filter_dynamic_range =
          std::min(max_area, fix16_range);  // int12 max value, pow(2,11)-1
    }

    format_dw_filter(param_.filter,
                     param_.quantizedFilter(),
                     null_scale,
                     filter_dynamic_range);

    uint32_t pool_limit = get_pool_cap();
    int kernel_h = param_.filter->shape().height();
    int kernel_rw =
        param_.filter->shape().width() +
        (param_.filter->shape().width() - 1) * (param_.dilations[0] - 1);
    image_reorder_ =
        (param_.dilations[0] > 1) &&
        (align_to_x(kernel_rw * channel * kernel_h, IMAGE_ALIGNMENT) >
         IMAGE_ALIGNMENT * pool_limit);

    DWconvArgs args = {0};
    args.bias_address = b_data;
    args.filter_address = param.quantizedFilter()->data<int16_t>();
    args.filter_scale_address = param_.quantizedFilter()->scale();
    args.kernel.width = param.filter->shape().height();
    args.kernel.height = param.filter->shape().width();
    args.kernel.stride_w = param.strides[0];
    args.kernel.stride_h = param.strides[1];
    args.image.address = input->data<void>();
    args.image.channels = input->shape().channel();
    args.image.height = input->shape().height();
    args.image.width = input->shape().width();
    args.image.pad_width = param.paddings[0];
    args.image.pad_height = param.paddings[1];
    args.image.scale_address = input->max();
    args.dilation = param_.dilations[0];
    args.output.address = output->data<void>();
    args.output.scale_address = output->max();
    args.out_width = param.output->shape().width();
    args.out_height = param.output->shape().height();
    args.sub_conv_num = 1;
    args.dilation = (param_.dilations[0] <= 1) ? 1 : param_.dilations[0];
    args.inplace.active_param.type = param_.activeParam.type;
    args.inplace.active_param.leaky_relu_factor =
        float_to_half(param_.activeParam.leaky_relu_factor);
    args.quant.dynamic_range =
        *(reinterpret_cast<uint16_t*>(&dynamic_range_fp16));
    args.quant.inv_dynamic_range =
        *(reinterpret_cast<uint32_t*>(&inv_dynamic_range));

    if (image_reorder_) {
      int reorder_img_height =
          param_.output->shape().height() * param_.filter->shape().height();
      int reorder_img_width =
          param_.output->shape().width() * param_.filter->shape().width();

      Shape shape_reorder(NHWC,
                          {1, reorder_img_height, reorder_img_width, channel});
      float16* reorder_data =
          reoder_input_.mutableData<float16>(FP16, shape_reorder);

      args.image.address = reorder_data;
      args.kernel.width = param_.filter->shape().width();
      args.kernel.height = param_.filter->shape().height();
      args.kernel.stride_w = param_.filter->shape().width();
      args.kernel.stride_h = param_.filter->shape().height();
      args.image.height = reorder_img_height;
      args.image.width = reorder_img_width;
      args.image.pad_width = 0;
      args.image.pad_height = 0;
      args.dilation = 1;
    }

    param_.args = args;
  }

  bool dispatch() {
    param_.input->syncToDevice();

    DWconvArgs& args = param_.args;

    if (image_reorder_) {
      int in_w = param_.input->shape().width();
      int in_h = param_.input->shape().height();
      int out_w = param_.output->shape().width();
      int out_h = param_.output->shape().height();
      int channel = param_.output->shape().channel();
      int kernel_w = param_.filter->shape().width();
      int kernel_h = param_.filter->shape().height();
      int stride_w = param_.strides[0];
      int stride_h = param_.strides[1];
      int pad_w = param_.paddings[0];
      int pad_h = param_.paddings[1];
      int dilation = param_.dilations[0];

      float16* orig_data = param_.input->data<float16>();
      float16* reorder_data = reoder_input_.data<float16>();

      for (int h_idx = 0; h_idx < out_h; h_idx++) {
        for (int kh_idx = 0; kh_idx < kernel_h; kh_idx++) {
          int h_p = h_idx * stride_h + dilation * kh_idx - pad_h;
          for (int w_idx = 0; w_idx < out_w; w_idx++) {
            for (int kw_idx = 0; kw_idx < kernel_w; kw_idx++) {
              int w_p = w_idx * stride_w + dilation * kw_idx - pad_w;
              if (w_p >= 0 && h_p >= 0 && w_p < in_w && h_p < in_h) {
                int dst_addr =
                    (w_idx * kernel_w + kw_idx) * channel +
                    (h_idx * kernel_h + kh_idx) * channel * out_w * kernel_w;
                int src_addr = w_p * channel + h_p * channel * in_w;
                memcpy(reorder_data + dst_addr,
                       orig_data + src_addr,
                       channel * sizeof(float16));
              }
            }
          }
        }
      }
      reoder_input_.flush();
    }

    if (param_.re_assign) {
      float16* scale_data = scale_bias_.data<float16>();
      int channel = param_.output->shape().channel();
      for (int i = 0; i < align_repeat_; i++) {
        int offset = channel * align_repeat_ + i * channel;
        memcpy(scale_data + offset,
               param_.scale()->data<float16>(),
               param_.scale()->memorySize());
      }
      scale_bias_.flush();
    }
    return compute_fpga_dwconv(param_.args) == 0;
  }

  DepthwiseConvParam& param() { return param_; }

 private:
  DepthwiseConvParam param_;
  Tensor scale_bias_;
  int align_repeat_ = 1;
  bool image_reorder_ = false;
  Tensor reoder_input_;
};

}  // namespace zynqmp
}  // namespace paddle
