// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/xpu/im2sequence_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

inline int Im2SeqOutputSize(
    int input_size, int filter_size, int padding_0, int padding_1, int stride) {
  const int output_size =
      (input_size + padding_0 + padding_1 - filter_size) / stride + 1;
  return output_size;
}

void Im2SequenceCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  auto x_dims = param.X->dims();

  int batch = x_dims[0];
  int channel = x_dims[1];
  int height = x_dims[2];
  int width = x_dims[3];
  int kernel_h = param.kernels[0];
  int kernel_w = param.kernels[1];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  int dilation_h = 1;
  int dilation_w = 1;
  int pad_h = param.paddings[0];
  int pad_w = param.paddings[1];

  int output_height =
      Im2SeqOutputSize(height, kernel_h, pad_h, pad_h, stride_h);
  int output_width = Im2SeqOutputSize(width, kernel_w, pad_w, pad_w, stride_w);

  std::vector<uint64_t> out_offset;
  out_offset.push_back(0);
  out_offset.push_back(output_height * output_width);

  for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
    int r = xdnn::im2col_ocf(
        ctx.GetRawContext(), /* context */
        channel,
        height,
        width,
        kernel_h,
        kernel_w,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
        dilation_h,
        dilation_w,
        param.X->data<float>() + batch_idx * channel * height * width,
        param.Out->mutable_data<float>(TARGET(kXPU)) +
            batch_idx * output_height * output_width * channel * kernel_h *
                kernel_w);
    CHECK_EQ(r, 0);
  }
  auto lod = param.Out->mutable_lod();
  lod->resize(1);
  (*lod)[0] = out_offset;
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(im2sequence,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::Im2SequenceCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
