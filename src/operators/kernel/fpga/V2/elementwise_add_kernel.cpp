/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#ifdef ELEMENTWISEADD_OP

#include "operators/kernel/elementwise_add_kernel.h"

#include <string>
#include "fpga/V2/api.h"

namespace paddle_mobile {
namespace operators {

template <>
bool ElementwiseAddKernel<FPGA, float>::Init(ElementwiseAddParam<FPGA> *param) {
  auto *input_y = const_cast<LoDTensor *>(param->InputY());
  auto *out = param->Out();
  if (input_y->type() != type_id<float>()) {
    paddle_mobile::fpga::ActivationType activation_enable =
        paddle_mobile::fpga::NONE;
    int16_t leaky_relu_negative_slope = 0;
    auto *input_x = const_cast<LoDTensor *>(param->InputX());
    auto input_x_ptr = input_x->data<half>();
    auto input_y_ptr = input_y->data<half>();
    fpga::format_ofm(out);
    auto out_ptr = out->mutable_data<half>();
    float Si_1 = input_x->scale[0];
    float Si_2 = input_y->scale[0];
    float So = out->scale[0];
    float C1 = Si_1 / So;
    float C2 = Si_2 / So;
    fpga::EWAddArgs ewaddArgs = {0};
    ewaddArgs.output.activation.activation_type = activation_enable;
    ewaddArgs.output.activation.leaky_relu_negative_slope =
        leaky_relu_negative_slope;
    ewaddArgs.const0 = fpga::fp32_2_fp16(C1);
    ewaddArgs.const1 = fpga::fp32_2_fp16(C2);
    ewaddArgs.image0.address = input_x_ptr;
    ewaddArgs.image0.channels = (uint32_t)input_x->dims()[1];
    ewaddArgs.image0.scale_address = input_x->scale;
    ewaddArgs.image0.height = (uint32_t)input_x->dims()[2];
    ewaddArgs.image0.width = (uint32_t)input_x->dims()[3];
    ewaddArgs.image0.pad_height = 0;
    ewaddArgs.image0.pad_width = 0;
    ewaddArgs.image1.address = input_y_ptr;
    ewaddArgs.image1.channels = (uint32_t)input_y->dims()[1];
    ewaddArgs.image1.scale_address = input_y->scale;
    ewaddArgs.image1.height = (uint32_t)input_y->dims()[2];
    ewaddArgs.image1.width = (uint32_t)input_y->dims()[3];
    ewaddArgs.image1.pad_height = 0;
    ewaddArgs.image1.pad_width = 0;
    ewaddArgs.output.scale_address = out->scale;
    ewaddArgs.output.address = out_ptr;
    fpga::expand_EW_arg(&ewaddArgs);
    param->SetFpgaArgs(ewaddArgs);
  } else {
    param->float_input_x.Resize(param->InputX()->dims());
    param->float_input_x.init(type_id<float>().hash_code());
    fpga::format_ofm(&(param->float_input_x));

    param->float_out.Resize(param->InputX()->dims());
    param->float_out.mutable_data<float>(param->InputX()->dims());
    fpga::format_ofm(&(param->float_out));

    fpga::format_ofm(out);
  }
  return true;
}

inline void ElementwiseAddCompute(const ElementwiseAddParam<FPGA> &param) {
  auto input_x = param.float_input_x;
  auto input_y = param.InputY();
  auto Out = param.float_out;
  int axis = param.Axis();

  const auto &x_dims = input_x.dims();
  const auto &y_dims = input_y->dims();
  /// axis = -1 represent the last dimensions.
  axis = (axis == -1 ? x_dims.size() - y_dims.size() : axis);
  size_t batch = 1;
  size_t channels = 1;
  size_t elementwise_num = 1;
  for (int i = 0; i < axis; ++i) {
    batch *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    channels *= y_dims[i];
  }
  for (int i = y_dims.size() + axis; i < x_dims.size(); ++i) {
    elementwise_num *= x_dims[i];
  }
  const float *bias_data = input_y->data<float>();
  const float *input_data = input_x.data<float>();
  float *output_data = Out.mutable_data<float>();

  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      size_t offset = (i * channels + j) * elementwise_num;
      const float *input = input_data + offset;
      const float bias = bias_data[j];
      float *output = output_data + offset;
      // DLOG << "output address: "<< output;
      for (int k = 0; k < elementwise_num; ++k) {
        output[k] = input[k] + bias;
        // DLOG << "output[" << k << "]= " << output[k] ;
      }
    }
  }
}
template <>
void ElementwiseAddKernel<FPGA, float>::Compute(
    const ElementwiseAddParam<FPGA> &param) {
  auto input_y = const_cast<LoDTensor *>(param.InputY());
  if (input_y->type() != type_id<float>()) {
    fpga::ComputeFpgaEWAdd(param.FpgaArgs());
  } else {
    auto input_x = const_cast<LoDTensor *>(param.InputX());
    auto intput_x_float = const_cast<Tensor *>(&(param.float_input_x));
    fpga::BypassArgs args = {fpga::DATA_TYPE_FP16};
    args.input_data_type = fpga::DATA_TYPE_FP16;
    args.output_data_type = fpga::DATA_TYPE_FP32;
    args.input_layout_type = fpga::LAYOUT_CHW;
    args.output_layout_type = fpga::LAYOUT_HWC;
    args.image.address = input_x->data<half>();
    args.image.channels = (uint32_t)(input_x->fpga_data_num);
    args.image.height = 1;
    args.image.width = 1;
    args.image.pad_height = 0;
    args.image.pad_width = 0;
    args.output.address = intput_x_float->data<float>();
    args.output.scale_address = intput_x_float->scale;

    // fpga::fpga_flush(input_x->data<half>(),input_x->fpga_data_num *
    // sizeof(half));
    fpga::PerformBypass(args);
    fpga::fpga_invalidate(args.output.address,
                          input_x->fpga_data_num * sizeof(float));

    // just for test
    /*    {
           static int cnt = 0;
           if(cnt == 0){
               std::string str= "first_bypass_data";
               float rslt = 0.0f;
               fpga::savefile(str, args.output.address, input_x->fpga_data_num,
       rslt); cnt++;
           }
       }*/
    ElementwiseAddCompute(param);

    auto out_float = const_cast<Tensor *>(&(param.float_out));
    DLOG << "out float: " << out_float->data<float>();
    fpga::fpga_flush(out_float->data<float>(),
                     input_x->fpga_data_num * sizeof(float));
    // just for test
    /*{
       static int cnt = 0;
       if(cnt == 0){
           std::string str= "ew_output_data";
           float rslt = 0.0f;

           fpga::savefile(str, out_float->data<float>(), input_x->fpga_data_num,
   rslt); cnt++;
       }
   }*/
    auto Out = param.Out();
    args.input_data_type = fpga::DATA_TYPE_FP32;
    args.output_data_type = fpga::DATA_TYPE_FP16;
    args.input_layout_type = fpga::LAYOUT_CHW;
    args.output_layout_type = fpga::LAYOUT_HWC;
    args.image.address = out_float->data<float>();
    args.image.channels = (uint32_t)(input_x->fpga_data_num);
    args.image.height = 1;
    args.image.width = 1;
    args.image.pad_height = 0;
    args.image.pad_width = 0;
    args.output.address = Out->data<half>();
    args.output.scale_address = Out->scale;
    fpga::PerformBypass(args);
  }
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
