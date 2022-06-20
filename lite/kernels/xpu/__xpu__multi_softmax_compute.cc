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

#include "lite/kernels/xpu/__xpu__multi_softmax_compute.h"
#include <vector>
#include "lite/backends/host/math/split.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void MultiSoftmaxCompute::PrepareForRun() {
  auto& param = this->template Param<param_t>();
  xpu_lod_guard_ =
      TargetWrapperXPU::MallocScratchPad(param.lod.size() * sizeof(int));
  XPU_CALL(xpu_memcpy(reinterpret_cast<int*>(xpu_lod_guard_->addr_),
                      param.lod.data(),
                      param.lod.size() * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_wait(TargetWrapperXPU::get_xpu_stream()));
  query_lod = {param.lod.data(),
               static_cast<int>(param.lod.size()),
               reinterpret_cast<int*>(xpu_lod_guard_->addr_)};
}

void MultiSoftmaxCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto lod = param.lod;
  CHECK(param.concat_output != nullptr);
  auto in_dim = param.input->dims();
  std::vector<int> x_shape;
  for (size_t i = 0; i < in_dim.size(); i++) {
    x_shape.push_back(in_dim[i]);
  }
  int r = xdnn::sequence_softmax<float>(
      ctx.GetRawContext(),
      param.input->data<float>(),
      param.concat_output->mutable_data<float>(TARGET(kXPU)),
      x_shape,
      1,
      query_lod);
  CHECK_EQ(r, 0);

  std::vector<float> cpu_concat_out(param.input->numel(), 0);
  lite::TargetWrapperXPU::MemcpySync(cpu_concat_out.data(),
                                     param.concat_output->data<float>(),
                                     cpu_concat_out.size() * sizeof(float),
                                     IoDirection::DtoH);
  lite::host::math::split(cpu_concat_out.data(), param.output, 1, x_shape);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(__xpu__multi_softmax,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::MultiSoftmaxCompute,
                     def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("ConcatOut", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
