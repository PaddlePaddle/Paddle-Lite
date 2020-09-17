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

#include "lite/kernels/xpu/sequence_pool_compute.h"
#include <string>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUSequencePoolCompute::PrepareForRun() {
  lod_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(
      XPU_MAX_LOD_SIZE * sizeof(int), false /* use_l3 */);
  lod_cpu.reset(new int[XPU_MAX_LOD_SIZE]);
}

void XPUSequencePoolCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* in = param.X;
  auto* out = param.Out;
  std::string pool_type_str = param.pool_type;

  auto dims = in->dims();
  auto lod = in->lod();
  dims[0] = lod[0].size() - 1;

  xdnn::Pooling_t pool_type = xdnn::Pooling_t::MAX_WITHOUT_INDEX;
  if (pool_type_str == "MAX") {
  } else if (pool_type_str == "SUM") {
    pool_type = xdnn::Pooling_t::SUM;
  } else if (pool_type_str == "LAST") {
    pool_type = xdnn::Pooling_t::LAST;
  } else {
    CHECK(false);
  }

  int num_seq = out->dims()[0];
  int dim = out->numel() / num_seq;

  auto in_lod = in->lod()[0];
  for (size_t i = 0; i < in_lod.size(); ++i) {
    lod_cpu[i] = in_lod[i];
  }
  int* lod_xpu = reinterpret_cast<int*>(lod_xpu_guard_->addr_);
  XPU_CALL(xpu_memcpy(lod_xpu,
                      lod_cpu.get(),
                      in_lod.size() * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));

  int r =
      xdnn::sequence_pooling_forward(ctx.GetRawContext(),
                                     pool_type,
                                     num_seq,
                                     lod_xpu,
                                     dim,
                                     in->data<float>(),
                                     nullptr /* index */,
                                     out->mutable_data<float>(TARGET(kXPU)));
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_pool,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::XPUSequencePoolCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("MaxIndex", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
