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
  lod_cpu.reset(new int[XPU_MAX_LOD_SIZE]);
}

void XPUSequencePoolCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* in = param.X;
  auto* out = param.Out;
  float pad_value = param.pad_value;
  std::string pool_type_str = param.pool_type;

  int num_seq = out->dims()[0];
  int dim = out->numel() / num_seq;

  auto in_lod = in->lod()[0];
  for (size_t i = 0; i < in_lod.size(); ++i) {
    lod_cpu[i] = in_lod[i];
  }
  int lod_len = in_lod.size();
  int r = 0;
  if (pool_type_str == "MAX") {
    r = xdnn::sequence_max_pool<float, int>(
        ctx.GetRawContext(),
        in->data<float>(),
        out->mutable_data<float>(TARGET(kXPU)),
        {lod_cpu.get(), lod_len, nullptr},
        num_seq,
        dim,
        pad_value,
        nullptr);
  } else if (pool_type_str == "SUM") {
    r = xdnn::sequence_sum_pool<float, int>(
        ctx.GetRawContext(),
        in->data<float>(),
        out->mutable_data<float>(TARGET(kXPU)),
        {lod_cpu.get(), lod_len, nullptr},
        num_seq,
        dim,
        pad_value);
  } else if (pool_type_str == "LAST") {
    r = xdnn::sequence_last_pool<float, int>(
        ctx.GetRawContext(),
        in->data<float>(),
        out->mutable_data<float>(TARGET(kXPU)),
        {lod_cpu.get(), lod_len, nullptr},
        num_seq,
        dim,
        pad_value);
  } else if (pool_type_str == "FIRST") {
    r = xdnn::sequence_first_pool<float, int>(
        ctx.GetRawContext(),
        in->data<float>(),
        out->mutable_data<float>(TARGET(kXPU)),
        {lod_cpu.get(), lod_len, nullptr},
        num_seq,
        dim,
        pad_value);
  } else {
    CHECK(false) << " unsupported pool_type_str: " << pool_type_str;
  }

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
