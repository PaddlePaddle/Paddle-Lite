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

#ifdef SOFTMAX_OP

#include "operators/kernel/softmax_kernel.h"
#include "operators/kernel/central-arm-func/softmax_arm_func.h"
namespace paddle_mobile {
namespace operators {

template <>
bool SoftmaxKernel<FPGA, float>::Init(SoftmaxParam<FPGA> *param) {
  auto input = const_cast<Tensor *>(param->InputX());
  auto input_ptr = input->data<float>();
  auto float_input = new Tensor;
  float_input->mutable_data<float>({1, input->dims()[1]});
  fpga::format_fp32_ofm(float_input, 8);

  fpga::BypassArgs args = {fpga::DATA_TYPE_FP16};
  args.input_layout_type = fpga::LAYOUT_HWC;
  args.output_layout_type = fpga::LAYOUT_CHW;
  args.input_data_type = fpga::DATA_TYPE_FP16;
  args.output_data_type = fpga::DATA_TYPE_FP32;
  args.image.address = input_ptr;
  args.image.height = 1;
  args.image.width = 1;
  args.image.channels = (uint32_t)input->dims()[1];
  args.output.address = float_input->data<float>();
  args.output.scale_address = float_input->scale;
  param->SetFloatInput(float_input);
  param->SetFpgaArgs(args);
  return true;
}

template <>
void SoftmaxKernel<FPGA, float>::Compute(const SoftmaxParam<FPGA> &param) {
  Tensor *in_x = param.FloatInput();
  Tensor *out = param.Out();

  fpga::PerformBypass(param.FpgaArgs());
  fpga::fpga_invalidate(
      (void *)in_x->data<float>(),                           // NOLINT
      fpga::get_aligned_channel_num((int)in_x->dims()[1]) *  // NOLINT
          sizeof(float));
  math::SoftmaxFuntor<CPU, float>()(in_x, out);
  fpga::fpga_flush(out->data<float>(), out->memory_size());
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
