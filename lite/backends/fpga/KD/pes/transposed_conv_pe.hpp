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
#include <vector>

#include "lite/backends/fpga/KD/llapi/zynqmp_api.h"
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/concat_pe.hpp"
#include "lite/backends/fpga/KD/pes/conv_pe.hpp"
#include "lite/backends/fpga/KD/pes/elementwise_add_pe.hpp"
#include "lite/backends/fpga/KD/pes/scale_pe.hpp"
#include "lite/backends/fpga/KD/pes/split_pe.hpp"
#include "lite/backends/fpga/KD/pes/transposed_conv_process.hpp"

namespace paddle {
namespace zynqmp {

class TransposedConvPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    int kernel_width = param_.filter->shape().width();
    int kernel_height = param_.filter->shape().height();
    int stride_width = param_.strides[0];
    int padding_width = param_.paddings[0];

    if (kernel_width % stride_width == 0) {
      sub_filter_ena_ = true;
    } else {
      sub_filter_ena_ = false;
    }
    // pad inputs on ZU3 devices
    if (DLEngine::get_instance().isZU3()) {
      sub_filter_ena_ = false;
    }

    ConvParam& conv_param = pe_.param();
    convert_cnhw_to_nchw(param_.filter, &filter_);
    inverse_filter(&filter_);

    if (sub_filter_ena_) {
      omit_size_ = deconv_get_omit(stride_width, kernel_width, padding_width);
      fill_sub_filters(&param_, &filter_);
      conv_param = const_cast<ConvParam&>(param_);
      conv_param.deconv = true;
      conv_param.activeParam.type = param_.activeParam.type;
    } else {
      Shape& input_shape = param_.input->shape();
      int padded_height = input_shape.height() +
                          (input_shape.height() - 1) * (param_.strides[0] - 1);
      int padded_width = input_shape.width() +
                         (input_shape.width() - 1) * (param_.strides[1] - 1);
      Shape padded_shape(NCHW,
                         {input_shape.num(),
                          input_shape.channel(),
                          padded_height,
                          padded_width});

      int ph = param_.filter->shape().height() - param_.paddings[0] - 1;
      int pw = param_.filter->shape().width() - param_.paddings[1] - 1;

      padded_input_.mutableData<float16>(FP16, padded_shape);
      conv_param.input = &padded_input_;
      conv_param.output = param_.output;
      conv_param.filter = &filter_;
      conv_param.strides = {1, 1};
      conv_param.paddings = {ph, pw};
      conv_param.kernelSize = {kernel_height, kernel_width};
      conv_param.dilations = {1, 1};
      conv_param.deconv = false;
      conv_param.activeParam.type = param_.activeParam.type;
      conv_param.scale()->mutableData<float>(FP32, param_.scale()->shape());
      conv_param.scale()->copyFrom(param_.scale());
      conv_param.bias()->mutableData<float>(FP32, param_.bias()->shape());
      conv_param.bias()->copyFrom(param_.bias());
    }
    pe_.init();
    pe_.apply();
  }

  template <typename T>
  void pad_input() {
    param_.input->syncToCPU();
    T* input_data = param_.input->data<T>();
    int channel = param_.input->shape().channel();
    int in_wc = param_.input->shape().width() * channel;
    int o_wc = padded_input_.shape().width() * channel;

    T* data = padded_input_.data<T>();
    int oh = param_.input->shape().height();
    int ow = param_.input->shape().width();
    memset(data, 0, padded_input_.memorySize());

    for (int h = 0; h < oh; h++) {
      for (int w = 0; w < ow; w++) {
        T* src = input_data + h * in_wc + w * channel;
        T* dst = data + (h)*param_.strides[0] * o_wc +
                 (w) * (param_.strides[1]) * channel;
        memcpy(dst, src, channel * sizeof(T));
      }
    }

    padded_input_.flush();
    padded_input_.copyScaleFrom(param_.input);
  }

  bool dispatch() {
    if (sub_filter_ena_ == false) {
      pad_input<float16>();
    }

    bool ret = pe_.dispatch();
    if (sub_filter_ena_ == true && ret == true) {
      int off_addr = omit_size_ * param_.output->shape().width() *
                     param_.output->shape().channel();
      param_.output->unalignImage();
      param_.output->setOffset(off_addr);
      float scale = 0.0;
      for (auto conv_param : param_.splitParams()) {
        scale = std::max(scale, conv_param->output.scale()[0]);
      }
      param_.output->scale()[0] = scale;
      param_.output->scale()[1] = 1.0f / scale;
    }
    return ret;
  }

  ConvParam& param() { return param_; }

 private:
  ConvParam param_;
  ConvPE pe_;
  bool sub_filter_ena_;
  int omit_size_;
  Tensor padded_input_;
  Tensor filter_;
};

}  // namespace zynqmp
}  // namespace paddle
