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

#ifdef TANH_OP

#include "operators/kernel/tanh_kernel.h"
#include <math.h>
namespace paddle_mobile {
namespace operators {

template <>
bool TanhKernel<FPGA, float>::Init(TanhParam<FPGA> *param) {
  auto input = const_cast<LoDTensor *>(param->InputX());
  DLOG << "input: " << input;
  auto input_ptr = input->data<half>();
  auto float_input = new LoDTensor;

  float_input->mutable_data<float>(
      {1, input->dims()[1], input->dims()[2], input->dims()[3]});
  fpga::format_fp32_ofm(float_input);

  fpga::BypassArgs args = {fpga::DATA_TYPE_FP16};
  args.input_layout_type = fpga::LAYOUT_HWC;
  args.output_layout_type = fpga::LAYOUT_CHW;
  args.input_data_type = fpga::DATA_TYPE_FP16;
  args.output_data_type = fpga::DATA_TYPE_FP32;
  args.image.address = input_ptr;
  args.image.height = (uint32_t)input->dims()[2];
  args.image.width = (uint32_t)input->dims()[3];
  args.image.channels = (uint32_t)input->dims()[1];
  args.output.address = float_input->data<float>();
  args.output.scale_address = float_input->scale;
  param->SetFloatInput(float_input);
  param->SetFpgaArgs(args);
  return true;
}

#define EXP_MAX_INPUT 40.0
template <typename T>
T Tanh(const T a) {
  T tmp = -2.0 * a;
  tmp = (tmp > EXP_MAX_INPUT) ? EXP_MAX_INPUT : tmp;
  return (2.0 / (1.0 + exp(tmp))) - 1.0;
}
template <typename T>
void tanhFuntor(Tensor *input, Tensor *output) {
  auto *input_ptr = input->data<T>();
  auto *output_ptr = output->mutable_data<T>();
  for (int i = 0; i < input->numel(); i++) {
    *(output_ptr + i) = Tanh<T>(*(input_ptr + i));
  }
}
template <>
void TanhKernel<FPGA, float>::Compute(const TanhParam<FPGA> &param) {
  Tensor *in_x = param.FloatInput();
  Tensor *out = param.Out();

  fpga::PerformBypass(param.FpgaArgs());
  fpga::fpga_invalidate(reinterpret_cast<void *>(in_x->data<float>()),
                        in_x->numel() * sizeof(float));
  tanhFuntor<float>(in_x, out);
  fpga::fpga_flush(out->data<float>(), out->memory_size());
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
