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

#include "lite/kernels/xpu/sequence_unpad_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SequenceUnpadCompute::PrepareForRun() {
  lod_xpu_guard_ = TargetWrapperXPU::MallocScratchPad(
      XPU_MAX_LOD_SIZE * sizeof(int), false /* use_l3 */);
  lod_cpu_.reserve(XPU_MAX_LOD_SIZE);
}

void SequenceUnpadCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto x_dims = param.X->dims();
  auto len_dims = param.Length->dims();

  // XXX(miaotianxiang): Target of tensor |Length| is |kHost|.
  auto* seq_len_ptr = param.Length->template data<int64_t>();
  int64_t batch_size = len_dims[0];
  std::vector<uint64_t> out_lod0(batch_size + 1, 0);
  for (int64_t i = 0; i < batch_size; ++i) {
    out_lod0[i + 1] = out_lod0[i] + seq_len_ptr[i];
  }
  paddle::lite::LoD out_lod;
  out_lod.push_back(out_lod0);

  int64_t out_dim0 = out_lod0.back();
  std::vector<int64_t> out_dims{out_dim0};
  if (x_dims.size() == 2) {
    out_dims.push_back(1);
  } else {
    for (size_t i = 2; i < x_dims.size(); ++i) {
      out_dims.push_back(x_dims[i]);
    }
  }
  param.Out->Resize(out_dims);
  param.Out->set_lod(out_lod);

  lod_cpu_ = {0};
  for (int64_t i = 0; i < batch_size; ++i) {
    int offset =
        lod_cpu_.back() + static_cast<int>(param.Length->data<int64_t>()[i]);
    lod_cpu_.push_back(offset);
  }
  lod_xpu_guard_->Reserve((batch_size + 1) * sizeof(int));
  TargetWrapperXPU::MemcpySync(lod_xpu_guard_->addr_,
                               lod_cpu_.data(),
                               (batch_size + 1) * sizeof(int),
                               IoDirection::HtoD);

  int dim = param.Out->numel() / out_dim0;
  int r = xdnn::sequence_unpad(
      ctx.GetRawContext(),                           /* ctx */
      param.X->data<float>(),                        /* pad_data */
      param.Out->mutable_data<float>(TARGET(kXPU)),  /* seq_data */
      reinterpret_cast<int*>(lod_xpu_guard_->addr_), /* sequence */
      param.X->dims()[1],                            /* pad_seq_len */
      batch_size,                                    /* batch_size */
      dim /* dim */);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_unpad,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SequenceUnpadCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Length",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
