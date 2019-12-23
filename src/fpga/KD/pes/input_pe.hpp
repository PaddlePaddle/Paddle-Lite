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

#include "../pe.hpp"
#include "../pe_params.hpp"
#include "../float16.hpp"
#include "../llapi/zynqmp_api.h"
#include "../preprocess_conf.hpp"
#include "common/common.h"
#include <fstream>
#include <iostream>

namespace paddle_mobile {
namespace zynqmp {

class InputPE : public PE {
 public:
  bool init() {
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);
    return true;
  }

  // bool dispatch() {
  //   fpga_reset();
  //   // std::cout << "InputPE dispatch \n";
  //   Tensor* input = param_.input;
  //   Tensor* output = param_.output;

  //   Tensor* src = input;
  //   input->flush();
  //   Tensor half_tensor;
  //   if (input->dataType() == DataType::FP32) {
  //     half_tensor.mutableData<void*>(DataType::FP16, input->shape());
  //     half_tensor.copyFrom(input);
  //     src = &half_tensor;
  //   }
  //   output->mutableData<void>();
  //   src->alignImage(output, true);
  //   return true;
  // }

   // bool init() {
    // Tensor* output = param_.output;
    // output->setAligned(true);
    // output->setDataLocation(Device);
    // std::cout << "init  param_.input->shape().num() :" << param_.input->shape().num()  << "," << param_.input->shape().channel() << std::endl;
    // std::cout << "init  param_.output->shape().num() :" << param_.output->shape().num() << "," << param_.output->shape().channel() << std::endl;
    // if (preprocess && param_.output->shape().channel() == 3) {
    //   zynqmp::ScaleParam& scale_param = scalePE_.param();
    //   scale_param.input = param_.output;
    //   scale_param.output = param_.output;
    //   std::cout << "init src channel :" <<   scale_param.input->shape().channel() << std::endl;

    //   zynqmp::Tensor* scale = new zynqmp::Tensor();
    //   zynqmp::Tensor* bias = new zynqmp::Tensor();
    //   zynqmp::Shape shape(zynqmp::N, {3});
    //   float* scale_data = scale->mutableData<float>(zynqmp::FP32, shape);
    //   float* bias_data = bias->mutableData<float>(zynqmp::FP32, shape);

    //   // scale_data[0] = 123.68;
    //   // scale_data[1] = 103.94;
    //   // scale_data[2] = 116.78;
    //   // bias_data[0] = -0.0131 * 123.68;
    //   // bias_data[1] = -0.0131 * 103.94;
    //   // bias_data[2] = -0.0131 * 116.78;

    //   scale_data[0] = 1;
    //   scale_data[1] = 1;
    //   scale_data[2] = 1;
    //   bias_data[0] = 1;
    //   bias_data[1] = 1;
    //   bias_data[2] = 1;
    //   scale->flush();
    //   bias->flush();

    //   scale_param.bias = bias;
    //   scale_param.scale = scale;
    //   std::cout << scale_data[0] << "," << scale_data[1] << "," << scale_data[2] << std::endl;
    //   std::cout << bias_data[0] << "," << bias_data[1] << "," << bias_data[2] << std::endl;

    //   scalePE_.init();
    //   scalePE_.apply();
    // }   
    // return true;
  // }

  bool dispatch() {
    auto time1 = time();
    fpga_reset();

    Tensor* input = param_.input;
    Tensor* output = param_.output;
    output->setAligned(true);
    output->setDataLocation(Device);  

    Tensor* src = input;
    input->flush();
    // input->saveToFile("input_0_", true);
    Tensor half_tensor;
    if (input->dataType() == DataType::FP32) {
      half_tensor.mutableData<void*>(DataType::FP16, input->shape());
      half_tensor.copyFrom(input);
      src = &half_tensor;
      // half_tensor.saveToFile("input_1_", true);
    }

    if (use_preprocess && param_.output->shape().channel() == 3) {
      src->alignImage();

      ScaleParam& scale_param = scalePE_.param();
      scale_param.input = src;
      scale_param.output = output;

      zynqmp::Tensor scale;
      zynqmp::Tensor bias;
      zynqmp::Shape shape(zynqmp::N, {3});
      float* scale_data = scale.mutableData<float>(zynqmp::FP32, shape);
      float* bias_data = bias.mutableData<float>(zynqmp::FP32, shape);

      scale_data[0] = preprocess_scale[0];
      scale_data[1] = preprocess_scale[1];
      scale_data[2] = preprocess_scale[2];
      bias_data[0] = -preprocess_scale[0] * preprocess_mean[0];
      bias_data[1] = -preprocess_scale[1] * preprocess_mean[1];
      bias_data[2] = -preprocess_scale[2] * preprocess_mean[2];

      scale.flush();
      bias.flush();

      scale_param.bias = &bias;
      scale_param.scale = &scale;

      scalePE_.init();
      scalePE_.apply();
      bool ret = scalePE_.dispatch();
      // output->alignImage();

    } else {
      output->mutableData<void>();
      src->alignImage(output, true);
      // output->saveToFile("input_2_", true);
    }
    
    return true;
  }

  InputParam& param() { return param_; }

 private:
  InputParam param_;
  ScalePE scalePE_;
  bool preprocess = true; 
};
}  // namespace zynqmp
}  // namespace paddle_mobile
