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
#include "operators/kernel/fetch_kernel.h"
namespace paddle_mobile {
namespace operators {

template <>
bool FetchKernel<FPGA, float>::Init(FetchParam<FPGA> *param) {
  auto input = const_cast<LoDTensor *>(param->InputX());
  int col = param->Col();
  DLOG << "col = " << col;
  auto output = &(param->Out()->at(col));
  output->init(type_id<float>().hash_code());
  output->Resize(input->dims());
  if (input->type() == type_id<float>()) {
    return true;
  }
  auto aligned_output = param->aligned_out;
  int outC = 1;
  int outW = 1;
  if (output->dims().size() == 4) {
    outC = output->dims()[1];
    outW = output->dims()[3];
  } else {  // 2
    outC = output->dims()[1];
  }
  int unalignedCW = outC * outW;
  int alignedCW = fpga::align_to_x(unalignedCW, IMAGE_ALIGNMENT);
  if (alignedCW != unalignedCW) {
    aligned_output.init(type_id<float>().hash_code());
    aligned_output.Resize(input->dims());
    fpga::format_fp32_ofm(&aligned_output);
  }
  return true;
}
void dealign(float *src, float *dst, int input_c, int input_h, int input_w) {
  int alignCW = paddle_mobile::fpga::align_to_x(input_c * input_w, 16);
  int dealignCW = input_c * input_w;
  for (int h = 0; h < input_h; ++h) {
    auto input_offset = h * alignCW;
    auto output_offset = h * dealignCW;
    memcpy((dst + output_offset), (src + input_offset),
           dealignCW * sizeof(float));
  }
}
template <>
void FetchKernel<FPGA, float>::Compute(const FetchParam<FPGA> &param) {
  auto input = const_cast<LoDTensor *>(param.InputX());
  int col = param.Col();
  auto output = &param.Out()->at(col);
  if (input->type() == type_id<float>()) {
    output->ShareDataWith(*input);
    return;
  }

  auto input_address = input->data<int8_t>();
  float Si = input->scale[0];
  auto aligned_ptr = const_cast<float *>(param.aligned_out.data<float>());
  auto outdata_ptr = output->data<float>();
  const int num_th = 32;
  fpga::fpga_invalidate(input_address, (input->fpga_data_num) * sizeof(int8_t));
  if (input->fpga_data_num < num_th) {
    for (int idx = 0; idx < product(input->dims()); ++idx) {
      outdata_ptr[idx] = input_address[idx] * Si;
    }
    fpga::fpga_flush(outdata_ptr, product(input->dims()) * sizeof(float));
    return;
  }
  for (int idx = 0; idx < input->fpga_data_num; ++idx) {
    aligned_ptr[idx] = input_address[idx] * Si;
  }
  int outC = 1;
  int outH = 1;
  int outW = 1;
  if (output->dims().size() == 4) {
    outC = output->dims()[1];
    outH = output->dims()[2];
    outW = output->dims()[3];
  } else {  // 2
    outC = output->dims()[1];
  }

  int unalignedCW = outC * outW;
  int alignedCW = fpga::align_to_x(unalignedCW, IMAGE_ALIGNMENT);
  if (unalignedCW != alignedCW) {
    dealign(aligned_ptr, outdata_ptr, outC, outH, outW);
    fpga::fpga_flush(outdata_ptr, outC * outH * outW * sizeof(float));
    return;
  }
  memcpy(outdata_ptr, aligned_ptr, outC * outH * outW * sizeof(float));
  fpga::fpga_flush(outdata_ptr, outC * outH * outW * sizeof(float));
}
template class FetchKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile
