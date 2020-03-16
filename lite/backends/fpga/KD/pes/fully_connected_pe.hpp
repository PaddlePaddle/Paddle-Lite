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

#include <math.h>
#include <cmath>
#include <vector>

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/conv_pe.hpp"
#include "lite/backends/fpga/KD/pes/conv_process.hpp"

namespace paddle {
namespace zynqmp {

class FullyConnectedPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    ConvParam& convParam_ = convPE_.param();
    Tensor* input = param_.input;
    convParam_.input = param_.input;
    convParam_.output = param_.output;
    convParam_.relu = param_.relu;
    // convParam_.activeParam.type = param_.activeParam.type;
    convParam_.groups = 1;
    convParam_.strides = {1, 1};
    convParam_.paddings = {0, 0};
    convParam_.kernelSize = {input->shape().width(), input->shape().height()};
    convParam_.dilations = {1, 1};

    int num = param_.filter->shape().channel();
    int chw = param_.filter->shape().num();
    int align = 32;
    int chw_aligned = ((chw + align - 1) / align) * align;

    int infer_num = 1;
    Shape in_shape(NCHW, {infer_num, chw_aligned, 1, 1});
    aligned_input_.mutableData<float16>(FP16, in_shape);
    convParam_.input = &aligned_input_;

    Shape out_shape(NCHW, {infer_num, num, 1, 1});
    aligned_output_.mutableData<float16>(FP16, out_shape);
    convParam_.output = &aligned_output_;

    int height = param_.input->shape().height();
    int width = param_.input->shape().width();
    // int filter_channel = chw / height / width;

    int channel = param_.output->shape().channel();
    Shape shape(NCHW, {num, chw_aligned, 1, 1});
    float* new_filter_data = conv_filter_.mutableData<float>(FP32, shape);
    float* filter_data = param_.filter->data<float>();

    memset(new_filter_data, 0, num * chw_aligned * sizeof(float));

    for (int i = 0; i < num; i++) {
      for (int j = 0; j < chw; j++) {
        float scale = filter_data[j * num + i];
        new_filter_data[i * chw_aligned + j] = scale;
      }
    }

    conv_filter_.flush();
    convParam_.filter = &conv_filter_;
    // param_.filter->saveToFile("param_filter", true);
    // conv_filter->saveToFile("conv_filter", true);
    // exit(-1);

    Shape sb_shape(N, {num});
    float* scale_data = convParam_.scale()->mutableData<float>(FP32, sb_shape);
    float* bias_data = convParam_.bias()->mutableData<float>(FP32, sb_shape);

    for (int i = 0; i < num; i++) {
      scale_data[i] = 1.0f;
      bias_data[i] = param_.bias->data<float>()[i];
    }
    // for (int i = 0; i < num; i++) {
    //   scale_data[i] = 1.0f;
    //   bias_data[i] = param_.bias->data<float>()[i];
    // }
    convParam_.scale()->flush();
    convParam_.bias()->flush();

    convPE_.init();
    convPE_.apply();
  }

  void cpu_compute() {
    int num = param_.filter->shape().channel();
    int chw = param_.filter->shape().num();

    float* filter_data = param_.filter->data<float>();
    float max = 0.0f;
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    float16* input_data = input->data<float16>();
    float16* output_data = output->data<float16>();

    for (int n = 0; n < input->shape().num(); n++) {
      float16* input_data = input->data<float16>() + n * chw;
      float16* output_data =
          output->data<float16>() + n * output->shape().channel();
      for (int i = 0; i < num; i++) {
        float sum = 0;
        float bias = param_.bias->data<float>()[i];
        for (int j = 0; j < chw; j++) {
          float scale = filter_data[j * num + i];
          float data = half_to_float(input_data[j]);
          sum += scale * data;
        }
        output_data[i] = float_to_half(sum + bias);
        if (max < output_data[i]) {
          max = output_data[i];
        }
      }
    }

    output->flush();
    output->scale()[0] = max / 127.0f;
    output->scale()[1] = 127.0f / max;
  }

  void cpu_compute1() {
    int num = conv_filter_.shape().num();
    int chw = conv_filter_.shape().channel();
    // chw = 336;

    float* filter_data = conv_filter_.data<float>();
    float max = 0.0f;
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    float16* input_data = input->data<float16>();
    float16* output_data = output->data<float16>();

    for (int n = 0; n < input->shape().num(); n++) {
      float16* input_data = input->data<float16>() + n * chw;
      float16* output_data =
          output->data<float16>() + n * output->shape().channel();
      for (int i = 0; i < num; i++) {
        float sum = 0;
        float bias = param_.bias->data<float>()[i];
        for (int j = 0; j < chw; j++) {
          float scale = filter_data[i * chw + j];
          float data = half_to_float(input_data[j]);
          sum += scale * data;
        }
        float value = sum + bias;
        if (std::isinf(value) || i > 321) {
          std::cout << "i:" << i << " sum:" << sum << " bias:" << bias
                    << std::endl;
          // exit(-1);
        }
        if (i > 321) {
          std::cout << "i:" << i << " sum:" << sum << " bias:" << bias
                    << std::endl;
          // exit(-1);
        }

        output_data[i] = float_to_half(value);
        if (max < value) {
          max = value;
        }
      }
    }

    output->flush();
    output->scale()[0] = max / 127.0f;
    output->scale()[1] = 127.0f / max;

    output->saveToFile("cpu_compute", true);
    // exit(-1);
  }

  void batch_to_w() {
    ConvParam& convParam_ = convPE_.param();

    int channel = param_.input->shape().channel();
    param_.input->invalidate();

    int remainder =
        aligned_input_.shape().channel() - param_.input->shape().channel();

    float max = 0;
    for (int n = 0; n < param_.input->shape().num(); n++) {
      memset(aligned_input_.data<float16>(),
             0,
             aligned_input_.shape().channel() * sizeof(float16));
      memcpy(
          aligned_input_.data<float16>() + n * aligned_input_.shape().channel(),
          param_.input->data<float16>() + n * channel,
          channel * sizeof(float16));
      aligned_input_.copyScaleFrom(param_.input);
      aligned_input_.flush();
    }

    convPE_.dispatch();
  }

  bool dispatch() {
    // batch_to_w();
    // return 1;
    // cpu_compute1();
    // return 1;

    // int num = param_.filter->shape().channel();
    // if (num == 2) {
    //   cpu_compute();
    //   return 1;
    // } else {
    // return convPE_.dispatch();
    // }
    ConvParam& convParam_ = convPE_.param();

    if (param_.input->shape().channel() == 321 &&
        param_.output->shape().channel() == 384) {
      // conv_filter_.saveToFile("conv_filter", true);
      // cpu_compute1();
      // return 1;
    }

    int channel = param_.input->shape().channel();
    param_.input->invalidate();

    int remainder =
        aligned_input_.shape().channel() - param_.input->shape().channel();

    float max = 0;
    for (int n = 0; n < param_.input->shape().num(); n++) {
      memset(aligned_input_.data<float16>(),
             0,
             aligned_input_.shape().channel() * sizeof(float16));
      memcpy(aligned_input_.data<float16>(),
             param_.input->data<float16>() + n * channel,
             channel * sizeof(float16));
      aligned_input_.copyScaleFrom(param_.input);
      aligned_input_.flush();

      if (param_.input->shape().channel() == 321 &&
          param_.output->shape().channel() == 384) {
        // aligned_input_.saveToFile("aligned_input_", true);
        // convParam_.filter->saveToFile("conv_filter", true);
      }

      convPE_.dispatch();
      aligned_output_.invalidate();
      if (param_.input->shape().num() == 230) {
        // aligned_output_.saveToFile("ao", true);
      }
      //
      float16* src = aligned_output_.data<float16>();
      float16* dst =
          param_.output->data<float16>() + n * param_.output->shape().channel();
      memcpy(dst, src, param_.output->shape().channel() * sizeof(float16));
      if (aligned_output_.scale()[0] > max) {
        max = aligned_output_.scale()[0];
      }
    }
    param_.output->flush();
    param_.output->scale()[0] = max / 127.0f;
    param_.output->scale()[1] = 127.0f / max;
    // param_.output->saveToFile("out", true);

    // exit(-1);
    // cpu_compute();

    // ConvParam& convParam_ = convPE_.param();
    // convParam_.scale()->saveToFile("scale", true);
    return true;
  }

  FullyConnectedParam& param() { return param_; }

 private:
  FullyConnectedParam param_;
  Tensor aligned_input_;
  Tensor aligned_output_;
  ConvPE convPE_;
  Tensor conv_filter_;
};
}  // namespace zynqmp
}  // namespace paddle
