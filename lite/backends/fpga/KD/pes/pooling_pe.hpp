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

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"

namespace paddle {
namespace zynqmp {

class PoolingPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    uint32_t k_height = 1;
    uint32_t k_width = 1;

    int input_c = input->shape().channel();
    if (param_.globalPooling) {
      k_width = input->shape().width();
      k_height = input->shape().height();
      param_.kernelSize[0] = k_height;
      param_.kernelSize[1] = k_width;
    } else {
      k_height = param_.kernelSize[0];
      k_width = param_.kernelSize[1];
    }

    int dynamic_range = (1 << 12) - 1;  // int13 max value, pow(2,12)-1
    float16 dynamic_range_fp16 = float_to_half(dynamic_range * 1.0);
    float inv_dynamic_range = 1.0 / dynamic_range;
    float kernel_reciprocal =
        (param_.type == PoolingType::MAX) ? 1.0 : (1.0 / (k_width * k_height));

    uint32_t pool_limit = 2 * get_pool_cap();
    use_cpu_ = align_to_x(k_width * input_c, IMAGE_ALIGNMENT) >
               IMAGE_ALIGNMENT * pool_limit;
    divide_pool_ = align_to_x(k_width * input_c, IMAGE_ALIGNMENT) * k_height >
                   IMAGE_ALIGNMENT * pool_limit;
    align_channel_ = (param_.paddings[0] * input_c) % IMAGE_ALIGNMENT != 0;

    PoolingArgs args = {0};
    args.mode = param_.type;
    args.image.channels = input->shape().channel();
    args.image.pad_height = param_.paddings[0];
    args.image.pad_width = param_.paddings[1];
    args.image.scale_address = input->max();
    args.output.scale_address = output->max();
    args.kernel.stride_h = param_.strides[0];
    args.kernel.stride_w = param_.strides[1];

    args.quant.dynamic_range =
        *(reinterpret_cast<uint16_t*>(&dynamic_range_fp16));
    args.quant.inv_dynamic_range =
        *(reinterpret_cast<uint32_t*>(&inv_dynamic_range));
    args.inplace.active_param.type = param_.activeParam.type;
    args.inplace.active_param.leaky_relu_factor =
        float_to_half(param_.activeParam.leaky_relu_factor);

    // global pool cases: BRAM can't contain one kernel
    if (divide_pool_) {
      float kw_reciprocal = 1.0 / k_width;
      float kh_reciprocal = 1.0 / k_height;

      Shape tmp_shape(NHWC, {1, input->shape().height(), 1, input_c});
      float16* mid_data = mid_out_.mutableData<float16>(FP16, tmp_shape);
      args.kernel_reciprocal = *(reinterpret_cast<uint32_t*>(&kw_reciprocal));
      args.image.width = input->shape().width();
      args.image.height = input->shape().height();
      args.kernel.height = 1;
      args.kernel.width = k_width;
      args.out_height = input->shape().height();
      args.out_width = 1;
      args.image.address = input->data<float16>();
      args.output.address = mid_out_.data<void>();
      args.output.scale_address = mid_out_.max();
      param_.poolingArgs = args;

      args.kernel_reciprocal = *(reinterpret_cast<uint32_t*>(&kh_reciprocal));
      args.image.width = 1;
      args.image.height = input->shape().height();
      args.kernel.height = k_height;
      args.kernel.width = 1;
      args.out_height = 1;
      args.out_width = 1;
      args.image.address = mid_data;
      args.output.address = output->mutableData<float16>();
      args.image.scale_address = mid_out_.max();
      args.output.scale_address = output->max();
      param_divide_.poolingArgs = args;
    } else {
      args.kernel_reciprocal =
          *(reinterpret_cast<uint32_t*>(&kernel_reciprocal));
      args.image.width = input->shape().width();
      args.image.height = input->shape().height();
      args.kernel.height = k_height;
      args.kernel.width = k_width;
      args.image.address = input->data<float16>();
      args.output.address = output->mutableData<float16>();
      args.out_height = output->shape().height();
      args.out_width = output->shape().width();

      if (align_channel_) {
        int align_c = align_to_x(input_c, IMAGE_ALIGNMENT);

        Shape in_shape(
            NHWC,
            {1, input->shape().height(), input->shape().width(), align_c});
        Shape out_shape(
            NHWC,
            {1, output->shape().height(), output->shape().width(), align_c});
        float16* tmp_in = input_tmp_.mutableData<float16>(FP16, in_shape);
        float16* tmp_out = output_tmp_.mutableData<float16>(FP16, out_shape);

        args.image.channels = align_c;
        args.image.address = tmp_in;
        args.output.address = tmp_out;
      }

      param_.poolingArgs = args;
    }
  }

  void compute() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    input->syncToCPU();

    Tensor float_input;
    float* image_addr = float_input.mutableData<float>(FP32, input->shape());
    float_input.copyFrom(input);
    float16* data_out = output->data<float16>();

    int image_height = input->shape().height();
    int image_width = input->shape().width();
    int image_channels = input->shape().channel();
    int image_pad_h = param_.paddings[0];
    int image_pad_w = param_.paddings[1];
    int kernel_height = param_.kernelSize[1];
    int kernel_width = param_.kernelSize[0];
    int kernel_step_h = param_.strides[0];
    int kernel_step_w = param_.strides[1];

    int pooled_height_ = output->shape().height();
    int pooled_width_ = output->shape().width();

    int kernel = kernel_height * kernel_width;

    float max = 0;

    for (int ph = 0; ph < pooled_height_; ++ph) {
      for (int pw = 0; pw < pooled_width_; ++pw) {
        int hstart = ph * kernel_step_h - image_pad_h;
        int wstart = pw * kernel_step_w - image_pad_w;
        int hend = std::min(hstart + kernel_height, image_height);
        int wend = std::min(wstart + kernel_width, image_width);
        hstart = std::max(hstart, 0);
        wstart = std::max(wstart, 0);

        kernel = (hend - hstart) * (wend - wstart);
        for (int c = 0; c < image_channels; ++c) {
          const int pool_index = (ph * pooled_width_ + pw) * image_channels + c;
          float sum = 0;

          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = (h * image_width + w) * image_channels + c;
              float value = image_addr[index];
              sum += value;
            }
          }

          float value = sum / kernel;
          if (value > max) {
            max = value;
          }
          data_out[pool_index] = float_to_half(value);
        }
      }
    }
    output->max()[0] = float_to_half(max);
    output->flush();
  }

  bool dispatch() {
    if (use_cpu_) {
      compute();
      return true;
    }
    int ret = 0;
    param_.input->syncToDevice();

    int channel = param_.input->shape().channel();
    int in_h = param_.input->shape().height();
    int in_w = param_.input->shape().width();
    int out_h = param_.output->shape().height();
    int out_w = param_.output->shape().width();
    int align_c = align_to_x(channel, IMAGE_ALIGNMENT);

    if (align_channel_) {
      float16* tmp_in = input_tmp_.data<float16>();
      float16* in_data = param_.input->data<float16>();
      for (int hw = 0; hw < in_h * in_w; hw++) {
        memcpy(tmp_in + hw * align_c,
               in_data + hw * channel,
               channel * sizeof(float16));
      }
      input_tmp_.flush();
    }

    ret = compute_fpga_pool(param_.poolingArgs);

    if (align_channel_) {
      float16* tmp_out = output_tmp_.data<float16>();
      float16* out_data = param_.output->data<float16>();
      output_tmp_.invalidate();
      for (int hw = 0; hw < out_h * out_w; hw++) {
        memcpy(out_data + hw * channel,
               tmp_out + hw * align_c,
               channel * sizeof(float16));
      }
    }

    if (ret == 0 && divide_pool_) {
      ret = compute_fpga_pool(param_divide_.poolingArgs);
    }
    return ret == 0;
  }

  PoolingParam& param() { return param_; }

 private:
  PoolingParam param_;
  PoolingParam param_divide_;
  Tensor mid_out_;
  Tensor input_tmp_;
  Tensor output_tmp_;
  bool use_cpu_ = false;
  bool divide_pool_ = false;
  bool align_channel_ = false;
};

}  // namespace zynqmp
}  // namespace paddle
