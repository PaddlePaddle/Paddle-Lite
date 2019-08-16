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
#ifdef POOL_OP

#include "operators/kernel/pool_kernel.h"

class PoolingArgs;
namespace paddle_mobile {
namespace operators {

template <>
bool PoolKernel<FPGA, float>::Init(PoolParam<FPGA> *param) {
  auto *input = const_cast<LoDTensor *>(param->Input());
  auto *output = param->Output();
  vector<int> ksize = param->Ksize();
  vector<int> strides = param->Strides();
  vector<int> paddings = param->Paddings();
  std::string pooling_type = param->PoolingType();

  if (input->type() == type_id<float>()) {
    int channels = input->dims()[1];
    int height = input->dims()[2];
    int width = input->dims()[3];
    int num = input->dims()[0];
    int out_width = (width + 2 * paddings[1] - ksize[1]) / strides[1] + 1;
    int out_height = (height + 2 * paddings[0] - ksize[0]) / strides[0] + 1;
    framework::DDim dim =
        framework::make_ddim({num, channels, out_height, out_width});
    output->mutable_data<float>(dim);
    return true;
  }

  auto input_ptr = input->data<int8_t>();
  fpga::format_ofm(output);
  auto output_ptr = output->mutable_data<int8_t>();
  float Si = input->scale[0];
  float So = output->scale[0];

  fpga::PoolingArgs poolArgs = {0};
  poolArgs.mode = pooling_type == "max" ? 0 : 1;  // max:0, avg:1
  poolArgs.kernel_reciprocal = fpga::fp32_2_fp16(
      float(1.0 / (ksize[0] * ksize[1]) * Si / So));  // NOLINT
  poolArgs.image.address = input_ptr;
  poolArgs.image.channels = (uint32_t)input->dims()[1];
  poolArgs.image.height = (uint32_t)input->dims()[2];
  poolArgs.image.width = (uint32_t)input->dims()[3];
  poolArgs.image.pad_height = (uint32_t)paddings[0];
  poolArgs.image.pad_width = (uint32_t)paddings[1];
  poolArgs.image.scale_address = input->scale;
  poolArgs.output.address = output_ptr;
  poolArgs.output.scale_address = output->scale;
  poolArgs.kernel.height = (uint32_t)ksize[0];
  poolArgs.kernel.width = (uint32_t)ksize[1];
  poolArgs.kernel.stride_h = (uint32_t)strides[0];
  poolArgs.kernel.stride_w = (uint32_t)strides[1];
  param->SetFpgaArgs(poolArgs);
  return true;
}

template <>
void PoolKernel<FPGA, float>::Compute(const PoolParam<FPGA> &param) {
  auto *input = const_cast<LoDTensor *>(param.Input());

  if (input->type() == type_id<float>()) {
    auto *output = param.Output();
    auto in = input->data<float>();
    auto N = input->dims()[0];
    output->Resize(
        {N, output->dims()[1], output->dims()[2], output->dims()[3]});
    auto len = output->numel();
    auto out = output->mutable_data<float>();
    int C = input->dims()[1], H = input->dims()[2],  // N = input->dims()[0],
        W = input->dims()[3];
    int HW = H * W, CHW = C * H * W, WC = W * C;

    for (int n = 0; n < N; n++) {
      for (int c = 0; c < C; c++) {
        out[n * C + c] = 0;
        for (int h = 0; h < H; h++) {
          for (int w = 0; w < W; w++) {
            out[n * C + c] += in[n * CHW + h * WC + w * C +
                                 c];  // in[n * CHW + c * HW + h * W + w]; //
          }
        }
        out[n * C + c] /= HW;
      }
    }
    return;
  }
  fpga::ComputeFpgaPool(param.FpgaArgs());
}
}  // namespace operators
}  // namespace paddle_mobile

#endif
