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

#ifdef ELEMENTWISEMUL_OP

#include "operators/kernel/elementwise_mul_kernel.h"
#include "operators/math/elementwise_op_function.h"

namespace paddle_mobile {
namespace operators {

template <typename T>
struct MulFunctor {
  inline T operator()(T a, T b) const { return a * b; }
};
template <>
bool ElementwiseMulKernel<FPGA, float>::Init(ElementwiseMulParam<FPGA> *param) {
  param->float_input_x.Resize(param->InputX()->dims());
  param->float_input_x.init(typeid(float));
  fpga::format_fp32_ofm(&(param->float_input_x));

  param->float_out.Resize(param->InputX()->dims());
  param->float_out.init(typeid(float));
  fpga::format_fp32_ofm(&(param->float_out));

  auto *out = param->Out();
  fpga::format_fp16_ofm(out);
  return true;
}

template <>
void ElementwiseMulKernel<FPGA, float>::Compute(
    const ElementwiseMulParam<FPGA> &param) {
  auto input_x = const_cast<LoDTensor *>(param.InputX());
  auto intput_x_float = const_cast<Tensor *>(&(param.float_input_x));
  // auto intput_x_32_ptr =
  // const_cast<float*>(param.float_input_x.data<float>());
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
  fpga::PerformBypass(args);
  fpga::fpga_invalidate(args.output.address,
                        input_x->fpga_data_num * sizeof(float));

  auto input_y = param.InputY();
  int axis = param.Axis();
  auto out_float = const_cast<Tensor *>(&(param.float_out));
  ElementwiseComputeEx<MulFunctor<float>, float>(
      intput_x_float, input_y, axis, MulFunctor<float>(), out_float);
  fpga::fpga_flush(out_float->data<float>(),
                   input_x->fpga_data_num * sizeof(float));

  Tensor *Out = param.Out();
  args.input_data_type = fpga::DATA_TYPE_FP32;
  args.output_data_type = fpga::DATA_TYPE_FP16;
  args.input_layout_type = fpga::LAYOUT_CHW;
  args.output_layout_type = fpga::LAYOUT_HWC;
  args.image.address = out_float->data<float>();
  args.image.channels = (uint32_t)(Out->fpga_data_num);
  args.image.height = 1;
  args.image.width = 1;
  args.image.pad_height = 0;
  args.image.pad_width = 0;
  args.output.address = Out->data<half>();
  args.output.scale_address = Out->scale;
  fpga::PerformBypass(args);
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
