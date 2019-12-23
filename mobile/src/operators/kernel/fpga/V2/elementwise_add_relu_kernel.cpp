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
#ifdef FUSION_ELEMENTWISEADDRELU_OP
#include "operators/kernel/elementwise_add_relu_kernel.h"
#include <math.h>

namespace paddle_mobile {
namespace operators {

template <>
bool ElementwiseAddReluKernel<FPGA, float>::Init(
    ElementwiseAddReluParam<FPGA> *param) {
  auto *input_x = const_cast<LoDTensor *>(param->InputX());
  auto *input_y = const_cast<LoDTensor *>(param->InputY());
  auto *out = param->Out();
  auto input_x_ptr = input_x->data<int8_t>();
  auto input_y_ptr = input_y->data<int8_t>();
  fpga::format_ofm(out);
  auto out_ptr = out->mutable_data<int8_t>();
  float Si_1 = input_x->scale[0];
  float Si_2 = input_y->scale[0];
  float So = out->scale[0];
  float C1 = Si_1 / So;
  float C2 = Si_2 / So;
  fpga::EWAddArgs ewaddArgs = {0};
  ewaddArgs.relu_enabled = 1;
  ewaddArgs.const0 = 1;
  ewaddArgs.const1 = 1;
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
  return true;
}

void ComputeCPUEWAddRelu(fpga::EWAddArgs ewaddArgs) {
  int inputc = ewaddArgs.image0.channels;
  int inputh = ewaddArgs.image0.height;
  int inputw = ewaddArgs.image0.width;
  float inScale0 =
      (reinterpret_cast<float *>(ewaddArgs.image0.scale_address))[0];
  float inScale1 =
      (reinterpret_cast<float *>(ewaddArgs.image1.scale_address))[0];
  float outScale =
      (reinterpret_cast<float *>(ewaddArgs.output.scale_address))[0];
  int8_t *inPtr0 = reinterpret_cast<int8_t *>(ewaddArgs.image0.address);
  int8_t *inPtr1 = reinterpret_cast<int8_t *>(ewaddArgs.image1.address);
  int8_t *outPtr = reinterpret_cast<int8_t *>(ewaddArgs.output.address);
  int datasize = inputc * inputh * inputw;
  float const0 = inScale0 / outScale;
  float const1 = inScale1 / outScale;
  fpga::fpga_invalidate(inPtr0, datasize * sizeof(int8_t));
  fpga::fpga_invalidate(inPtr1, datasize * sizeof(int8_t));
  for (int i = 0; i < datasize; i++) {
    float tmpF = inPtr0[i] * const0 + inPtr1[i] * const1;
    int tmpI = static_cast<int>(round(tmpF));
    outPtr[i] = (int8_t)((tmpI > 127 ? 127 : (tmpI < 0 ? 0 : tmpI)));
  }
  fpga::fpga_flush(outPtr, datasize * sizeof(int8_t));
}

template <>
void ElementwiseAddReluKernel<FPGA, float>::Compute(
    const ElementwiseAddReluParam<FPGA> &param) {
  // fpga::ComputeFpgaEWAdd(param.FpgaArgs());
  ComputeCPUEWAddRelu(param.FpgaArgs());
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
