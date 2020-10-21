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

#include <cstring>
#include <vector>

#include "lite/backends/fpga/KD/float16.hpp"
#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"

namespace paddle {
namespace zynqmp {

class NormPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  void apply() {
    inplace_args_.relu_enable = false;
    inplace_args_.power_enable = false;
    inplace_args_.normalize_enable = true;

    Shape& input_shape = param_.input->shape();

    norm_param_args_.channel = input_shape.channel();
    norm_param_args_.hight_width = input_shape.height() * input_shape.width();

    float16* mid_data =
        mid_out_.mutableData<float16>(FP16, param_.output->shape());

    bypass_args_.input_data_type = DATA_TYPE_FP16;
    bypass_args_.output_data_type = DATA_TYPE_FP16;
    bypass_args_.input_layout_type = LAYOUT_HWC;
    bypass_args_.output_layout_type = LAYOUT_HWC;
    bypass_args_.image.address = param_.input->data<void>();
    bypass_args_.image.scale_address = param_.input->scale();
    bypass_args_.image.channels = input_shape.channel();
    bypass_args_.image.height = input_shape.height();
    bypass_args_.image.width = input_shape.width();
    bypass_args_.output.address = mid_out_.data<void>();
    bypass_args_.output.scale_address = mid_out_.scale();

    norm_args_.input_image_address = mid_data;
    norm_args_.image_width = input_shape.width();
    norm_args_.image_height = input_shape.height();
    norm_args_.image_channel = input_shape.channel();
    norm_args_.output_image_address = param_.output->data<float>();
    norm_args_.output_scale_address =
        reinterpret_cast<uint32_t*>(param_.output->scale());
  }

  void cpuCompute() {
    Tensor input_float;
    Tensor float_out;
    input_float.mutableData<float>(FP32, param_.input->shape());
    float_out.mutableData<float>(FP32, param_.output->shape());

    // param_.input->syncToDevice();
    input_float.copyFrom(param_.input);
    input_float.syncToCPU();
    // input_float.saveToFile("normalize_", true);

    int channel = input_float.shape().channel();
    int height = input_float.shape().height();
    int width = input_float.shape().width();
    int cw = channel * width;

    Tensor* input = &input_float;
    float* input_ptr = input->data<float>();
    float* out_ptr = float_out.data<float>();

    int loop = height * width;
#pragma omp parallel for
    for (int i = 0; i < loop; i++) {
      float sum = param_.epsilon;
      for (int c = 0; c < channel; c++) {
        float value = input_ptr[i * channel + c];
        sum += value * value;
      }
      float norm = sqrtf(sum);
#pragma omp parallel for
      for (int c = 0; c < channel; c++) {
        out_ptr[i * channel + c] = input_ptr[i * channel + c] / norm;
      }
    }
    float_out.flush();
    // float_out.saveToFile("normalize_", true);
    param_.output->copyFrom(&float_out);
  }

  bool dispatch() {
    // cpuCompute();
    // std::cout << "FPGA normalize ---------------------" << std::endl;

    param_.input->syncToDevice();
    config_norm_param(norm_param_args_);
    inplace_args_.normalize_enable = true;
    config_inplace(inplace_args_);

    perform_bypass(bypass_args_);
    inplace_args_.normalize_enable = false;
    config_inplace(inplace_args_);
    compute_norm(norm_args_);

    return true;
  }

  NormParam& param() { return param_; }

 private:
  NormParam param_;
  Tensor mid_out_;
  InplaceArgs inplace_args_ = {0};
  NormalizeParameterArgs norm_param_args_ = {0};
  BypassArgs bypass_args_;

  NormalizeArgs norm_args_ = {0};
};

}  // namespace zynqmp
}  // namespace paddle
