// /* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. */

// #pragma once

// #include "../pe.hpp"
// #include "../pe_params.hpp"

// namespace paddle_mobile {
// namespace zynqmp {

// class ElementwiseMulPE : public PE {
//  public:
//   bool init() {
//     Tensor* output = param_.output;
//     output->setAligned(true);
//     output->setDataLocation(Device);
//     return true;
//   }

//  void apply() {
//     DepthwiseConvParam param;
//     Tensor* input = param_.input;
//     Tensor* output = param_.output;
//     int channel = output->shape().channel();
    
//     float16* filter_data = filter_.mutableData<float16>(FP16, param_.scale()->shape());
//     float16* scale_data = filter_.mutableData<float16>(FP16, param_.scale()->shape());
//     float16* b_data = bias_.mutableData<float16>(FP16, param_.bias()->shape());
    
//     DWconvArgs args = {0};
//     args.bias_address = b_data;
//     args.filter_address = filter_data;
//     args.kernel.width = 1;
//     args.kernel.height = 1;
//     args.kernel.stride_w = 1;
//     args.kernel.stride_h = 1;
//     args.image.address = input->data<void>();
//     args.image.channels = input->shape().channel();
//     args.image.height = input->shape().height();
//     args.image.width = input->shape().width();
//     args.image.pad_width = 1;
//     args.image.pad_height = 1;
//     args.image.scale_address = input->scale();
//     args.output.address = output->data<void>();
//     args.output.scale_address = output->scale();
//     args.out_width = param.output->shape().width();
//     args.out_height = param.output->shape().height();
//     args.sub_conv_num = 1;
//     param.args = args;

//     inplace_.relu_enable = param_.relu.enabled;
//     inplace_.power_enable = false;
//     inplace_.normalize_enable = false;
//   }

//   bool dispatch() {
//     param_.input->syncToDevice();
//     // if (param_.relu.enabled) {
//     //   inplace_.relu_enable = param_.relu.enabled;
//     //   config_inplace(inplace_);
//     // }
//     bool ret = compute_fpga_dwconv(param_.args) == 0;
//     // if (param_.relu.enabled) {
//     //   inplace_.relu_enable = false;
//     //   config_inplace(inplace_);
//     // }
//     return ret;
//   }

//   // ElementwiseMulParam& param() { return param_; }

//  private:
//   ElementwiseMulParam param_;
//   // Tensor bias_;
//   // Tensor filter_;
//   // InplaceArgs inplace_ = {0};

// };

// }  // namespace zynqmp
// }  // namespace paddle_mobile
