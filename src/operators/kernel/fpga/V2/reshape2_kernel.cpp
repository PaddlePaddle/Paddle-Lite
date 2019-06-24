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

#ifdef RESHAPE2_OP

#include "operators/kernel/reshape2_kernel.h"
#include "framework/ddim.h"

namespace paddle_mobile {
namespace operators {

template <>
bool Reshape2Kernel<FPGA, float>::Init(Reshape2Param<FPGA> *param) {
  auto input = const_cast<LoDTensor *>(param->InputX());
  auto output = param->Out();
  auto shape = param->Shape();

  auto num_in = framework::product(input->dims());
  auto num_shape = framework::product(framework::make_ddim(shape));
  PADDLE_MOBILE_ENFORCE(num_shape != 0, "0 index is not supported");

  for (int i = 0; i < shape.size(); i++) {
    if (shape[i] == -1) {
      shape[i] = static_cast<int>(-num_in / num_shape);
      break;
    }
  }
  output->Resize(framework::make_ddim(shape));
  output->set_type(input->type());
  fpga::format_ofm(output);
  DLOG << "input: " << input;
  DLOG << "output: " << output;

  return true;
}

void reshape(LoDTensor *input, LoDTensor *output) {
  // Subscript r means after reshape

  auto input_ptr = input->data<int8_t>();
  auto output_ptr = output->data<int8_t>();
  output->scale[0] = input->scale[0];
  output->scale[1] = input->scale[1];

  auto C = static_cast<int>(input->dims()[1]);
  auto H = static_cast<int>(input->dims()[2]);
  auto W = static_cast<int>(input->dims()[3]);
  auto Cr = static_cast<int>(output->dims()[1]);
  auto Hr = static_cast<int>(output->dims()[2]);
  auto Wr = static_cast<int>(output->dims()[3]);
  PADDLE_MOBILE_ENFORCE(C * H * W == Cr * Hr * Wr, "Dims don't match");
  auto WC = W * C;
  auto WC_align = fpga::align_to_x(WC, IMAGE_ALIGNMENT);
  auto HW = H * W;
  auto WCr = Wr * Cr;
  auto WCr_align = fpga::align_to_x(WCr, IMAGE_ALIGNMENT);
  auto HWr = Hr * Wr;

  fpga::fpga_invalidate(input_ptr, H * WC_align * sizeof(int8_t));

  int offset_align = 0;
  int offset_r = 0, offset_align_r = 0;
  int cr = 0, hr = 0, wr = 0;

  for (int h = 0; h < H; h++) {
    int offset0 = h * WC_align;
    for (int w = 0; w < W; w++) {
      int offset1 = w * C + offset0;
      for (int c = 0; c < C; c++) {
        offset_align = offset1 + c;
        offset_r = c * HW + h * W + w;
        cr = offset_r / HWr;
        hr = offset_r % HWr / Wr;
        wr = offset_r % Wr;
        offset_align_r = hr * WCr_align + wr * Cr + cr;
        output_ptr[offset_align_r] = input_ptr[offset_align];
      }
    }
  }

  fpga::fpga_flush(output_ptr, Hr * WCr_align * sizeof(int8_t));
}

template <>
void Reshape2Kernel<FPGA, float>::Compute(const Reshape2Param<FPGA> &param) {
  auto input = const_cast<LoDTensor *>(param.InputX());
  auto output = param.Out();
  auto shape = param.Shape();

  auto num_in = framework::product(input->dims());
  auto num_shape = framework::product(framework::make_ddim(shape));
  PADDLE_MOBILE_ENFORCE(num_shape != 0, "0 index is not supported");

  for (int i = 0; i < shape.size(); i++) {
    if (shape[i] == -1) {
      shape[i] = static_cast<int>(-num_in / num_shape);
      break;
    }
  }
  output->Resize(framework::make_ddim(shape));
  if (output->dims() == input->dims()) {
    DLOG << "No need to reshape";
    output->ShareDataWith(*input);
    framework::LoD lod = input->lod();
    output->set_lod(lod);
    output->scale[0] = input->scale[0];
    return;
  }

  reshape(input, output);
  //
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
