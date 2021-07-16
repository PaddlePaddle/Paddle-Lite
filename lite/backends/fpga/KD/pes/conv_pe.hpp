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

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
#include "lite/backends/fpga/KD/pes/concat_pe.hpp"
#include "lite/backends/fpga/KD/pes/conv_pe.hpp"
#include "lite/backends/fpga/KD/pes/conv_process.hpp"
#include "lite/backends/fpga/KD/pes/elementwise_add_pe.hpp"
#include "lite/backends/fpga/KD/pes/scale_pe.hpp"
#include "lite/backends/fpga/KD/pes/split_pe.hpp"

namespace paddle {
namespace zynqmp {

class ConvPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    if (param_.deconv == false) {
      split_axis = fill_split_arg(param_);
      pack_channel = split_axis == 2 && param_.splitParams().size() > 1;
      split_cpu_concat = split_axis == 0 && param_.cpu_concat;
      split_channel = split_axis == 1;
      for (auto conv_param : param_.splitParams()) {
        conv_param->args.inplace.active_param.type = param_.activeParam.type;
        conv_param->args.inplace.active_param.leaky_relu_factor =
            float_to_half(param_.activeParam.leaky_relu_factor);
      }

      if (pack_channel) {
        ConcatParam& concat_param = concatPE_.param();
        for (auto conv_param : param_.splitParams()) {
          concat_param.inputs.push_back(&conv_param->output);
        }
        concat_param.output = param_.output;
        concatPE_.init();
        concatPE_.apply();
      } else if (split_cpu_concat) {
        ConcatParam& concat_param = concatPE_.param();

        BasicConvParam* first = param_.splitParams().front();
        concat_param.inputs.push_back(&(first->output));

        BasicConvParam* last = param_.splitParams().back();
        concat_param.inputs.push_back(&(last->output));

        concat_param.output = param_.output;
        concatPE_.init();
        concatPE_.apply();
      }

      if (pack_channel || split_channel) {
        SplitParam& split_param = splitPE_.param();
        split_param.input = param_.input;
        for (auto conv_param : param_.splitParams()) {
          split_param.outputs.push_back(&conv_param->input);
        }
        splitPE_.init();
        splitPE_.apply();
      }
    }

    if (!use_cpu_) {
      param_.filter->releaseData();
    }
  }
  void cpu_compute() {
    Tensor* input = param_.input;
    Tensor* output = param_.output;
    input->syncToCPU();

    Tensor float_input;
    Tensor float_output;
    float* image_addr = float_input.mutableData<float>(FP32, input->shape());
    float_input.copyFrom(input);

    float* out = float_output.mutableData<float>(FP32, output->shape());
    int out_channel = output->shape().channel();
    int in_channel = input->shape().channel();

    float* filter_data = param_.filter->data<float>();
    float* mi = new float[in_channel];
    for (int i = 0; i < out_channel; i++) {
      float* image = image_addr;
      float* filter_ptr = filter_data + i * in_channel;
      float* out_ptr = mi;
#pragma omp parallel for
      for (int j = 0; j < in_channel; j++) {
        float value = image_addr[j] * filter_ptr[j];
        mi[j] = value;
      }

      float sum = 0;
      for (int j = 0; j < in_channel; j++) {
        sum += mi[j];
      }
      out[i] = sum;
    }
    delete[] mi;
    float_output.flush();
    output->flush();
    output->copyFrom(&float_output);
    output->invalidate();
  }

  bool dispatch() {
    if (use_cpu_) {
      cpu_compute();
      return true;
    }

    std::vector<BasicConvParam*>& params = param_.splitParams();

    if ((pack_channel || split_channel) && !param_.deconv) {
      splitPE_.dispatch();
    }

    int ret = 0;

    for (auto conv_param : params) {
      ret |= compute_fpga_conv_basic(conv_param->args);
    }

    if ((pack_channel || split_cpu_concat) && ret == 0 && !param_.deconv) {
      concatPE_.dispatch();
    }
    if (!split_channel && !param_.deconv) {
      float16 max_val = 0.0;

      /*
        The final result(multiple tensors concated by channel axis) is merged
        into a Tensor,
        but each channel's max value is calculated separately, we need to find a
        global max for this Tensor.
      */
      if (param_.wd_enable) {
        int cur_idx = param_.fuse_idx;
        if (cur_idx == param_.start_idx)
          max_val = 0;
        else if (cur_idx <= param_.end_idx)
          max_val = param_.output->max()[0];
      }

      for (auto conv_param : param_.splitParams()) {
        max_val = std::max(max_val, conv_param->output_max);
      }
      param_.output->max()[0] = max_val;
    }

    if (split_channel && ret == 0 && params.size() > 1) {
      ElementwiseAddParam& add_param = addPE_.param();
      add_param.inputs = {&params[0]->output, &params[1]->output};
      add_param.output = param_.output;
      addPE_.init();
      addPE_.apply();
      addPE_.dispatch();
    }
    return ret == 0;
  }

  ConvParam& param() { return param_; }

 private:
  bool use_cpu_ = false;
  bool pack_channel = false;
  bool split_cpu_concat = false;
  bool split_channel = false;
  ConvParam param_;
  ConcatPE concatPE_;
  SplitPE splitPE_;
  ElementwiseAddPE addPE_;
  int split_axis = 0;
};

}  // namespace zynqmp
}  // namespace paddle
