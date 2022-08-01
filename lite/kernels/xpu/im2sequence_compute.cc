// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

void Im2SequenceCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto x_dims = param.X->dims();

  int batch = x_dims[0];
  int channel = x_dims[1];
  int height = x_dims[2];
  int width = x_dims[3];
  int kernel_h = param.kernels[0];
  int kernel_w = param.kernels[1];

  std::vector<uint64_t> im_offset;
  im_offset.push_back(0);

  XPUScratchPadGuard xpu_x_nhwc_guard_ =
      TargetWrapperXPU::MallocScratchPad(param.X->numel() * sizeof(float));
  float* x_nhwc = reinterpret_cast<float*>(xpu_x_nhwc_guard_->addr_);

  XPUScratchPadGuard xpu_y_ofc_guard_ =
      TargetWrapperXPU::MallocScratchPad(param.Out->numel() * sizeof(float));
  float* y_ofc = reinterpret_cast<float*>(xpu_y_ofc_guard_->addr_);

  int r = xdnn::transpose<float>(ctx.GetRawContext(),
                                 param.X->data<float>(),
                                 x_nhwc,
                                 {batch, channel, height, width},
                                 {0, 2, 3, 1});

  CHECK_EQ(r, 0);
  r = xdnn::im2col<float>(ctx.GetRawContext(),
                          x_nhwc,
                          y_ofc,
                          batch,
                          channel,
                          height,
                          width,
                          param.kernels,
                          param.strides,
                          param.paddings,
                          {1, 1},
                          false);

  CHECK_EQ(r, 0);
  int output_imsize =
      param.Out->numel() / channel / (kernel_h * kernel_w) / batch;
  r = xdnn::transpose<float>(
      ctx.GetRawContext(),
      y_ofc,
      param.Out->mutable_data<float>(TARGET(kXPU)),
      {batch, output_imsize, kernel_h * kernel_w, channel},
      {0, 1, 3, 2});
  CHECK_EQ(r, 0);

  for (int im_id = 0; im_id < batch; im_id++) {
    im_offset.push_back(uint64_t((im_id + 1) * output_imsize));
  }
  auto lod = param.Out->mutable_lod();
  lod->resize(1);
  (*lod)[0] = im_offset;
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
