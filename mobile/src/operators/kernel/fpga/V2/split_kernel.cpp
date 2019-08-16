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

#ifdef SPLIT_OP

#include "operators/kernel/split_kernel.h"

namespace paddle_mobile {
namespace operators {
template <>
bool SplitKernel<FPGA, float>::Init(SplitParam<FPGA> *param) {
  auto *in = const_cast<LoDTensor *>(param->InputX());
  auto outs = param->Outs();
  auto sections = param->Sections();
  int axis = param->Axis();
  PADDLE_MOBILE_ENFORCE(axis == 1, "Only support split in channel dimension");
  PADDLE_MOBILE_ENFORCE(outs.size() == sections.size(),
                        "Output number should be equal to section number");
  auto image_num = (uint32_t)outs.size();
  auto images_out =
      reinterpret_cast<void **>(fpga::fpga_malloc(image_num * sizeof(void *)));
  auto scales_out = reinterpret_cast<float **>(
      fpga::fpga_malloc(image_num * sizeof(float *)));
  auto out_channels = reinterpret_cast<uint32_t *>(
      fpga::fpga_malloc(image_num * sizeof(uint32_t)));
  DLOG << "input: " << in;
  for (int i = 0; i < image_num; i++) {
    fpga::format_ofm(outs[i]);
    DLOG << "output: " << outs[i];
    images_out[i] = outs[i]->mutable_data<int8_t>();
    scales_out[i] = outs[i]->scale;
    out_channels[i] = (uint32_t)sections[i];
  }

  auto deleter = [](void *p) { fpga::fpga_free(p); };

  fpga::SplitArgs arg = {0};
  arg.image_num = image_num;
  arg.image_in = in->data<int8_t>();
  arg.scale_in = in->scale;
  arg.images_out = images_out;
  arg.scales_out = scales_out;
  arg.out_channel_nums = out_channels;
  arg.height = (uint32_t)in->dims()[2];
  arg.width = (uint32_t)in->dims()[3];
  arg.vector_split_space.push_back(
      std::shared_ptr<char>(reinterpret_cast<char *>(images_out), deleter));
  arg.vector_split_space.push_back(
      std::shared_ptr<char>(reinterpret_cast<char *>(scales_out), deleter));
  arg.vector_split_space.push_back(
      std::shared_ptr<char>(reinterpret_cast<char *>(out_channels), deleter));

  param->SetFpgaArgs(arg);
  return true;
}
template <>
void SplitKernel<FPGA, float>::Compute(const SplitParam<FPGA> &param) {
  fpga::ComputeFPGASplit(param.FpgaArgs());
}

}  // namespace operators
}  // namespace paddle_mobile
#endif
