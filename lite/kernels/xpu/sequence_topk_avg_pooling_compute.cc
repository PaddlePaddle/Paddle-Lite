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

#include "lite/kernels/xpu/sequence_topk_avg_pooling_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void SequenceTopkAvgPoolingCompute::PrepareForRun() {
  lod_xpu_guard_ =
      TargetWrapperXPU::MallocScratchPad(4 * XPU_MAX_LOD_SIZE * sizeof(int));
  row_lod_cpu.reset(new int[XPU_MAX_LOD_SIZE]);
  col_lod_cpu.reset(new int[XPU_MAX_LOD_SIZE]);
}

void SequenceTopkAvgPoolingCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();

  auto* in = param.X;
  auto* row = param.ROW;
  auto* col = param.COLUMN;
  auto* out = param.Out;
  auto* pos = param.pos;

  auto channel_num = param.channel_num;
  auto topks = param.topks;
  auto k_num = topks.size();
  auto max_k = topks[topks.size() - 1];

  auto row_lod = row->lod()[0];
  auto col_lod = col->lod()[0];
  int batch_size = row_lod.size() - 1;
  int pos_total_size = row_lod[batch_size] * channel_num * max_k;
  std::vector<int64_t> vec_pos_shape;
  vec_pos_shape.push_back(pos_total_size);
  pos->Resize(vec_pos_shape);
  auto pos_data = pos->mutable_data<int>(TARGET(kXPU));

  int offset = 0;
  std::vector<uint64_t> vec_out_lod;
  vec_out_lod.reserve(batch_size + 1);
  for (int i = 0; i <= batch_size; ++i) {
    offset = row_lod[i];
    vec_out_lod.push_back(offset);
  }
  LoD lod_temp;
  lod_temp.push_back(vec_out_lod);
  out->set_lod(lod_temp);

  auto in_data = in->data<float>();
  auto out_data = out->mutable_data<float>(TARGET(kXPU));

  int* row_lod_xpu = reinterpret_cast<int*>(lod_xpu_guard_->addr_);
  int* col_lod_xpu = row_lod_xpu + row_lod.size();
  int* topks_xpu = col_lod_xpu + col_lod.size();
  for (int i = 0; i < row_lod.size(); ++i) {
    row_lod_cpu[i] = row_lod[i];
  }
  for (int i = 0; i < col_lod.size(); ++i) {
    col_lod_cpu[i] = col_lod[i];
  }
  XPU_CALL(xpu_memcpy(row_lod_xpu,
                      row_lod_cpu.get(),
                      row_lod.size() * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(col_lod_xpu,
                      col_lod_cpu.get(),
                      col_lod.size() * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  XPU_CALL(xpu_memcpy(topks_xpu,
                      topks.data(),
                      topks.size() * sizeof(int),
                      XPUMemcpyKind::XPU_HOST_TO_DEVICE));
  int r = xdnn::sequence_topk_avg_pooling<float, int>(
      ctx.GetRawContext(),
      in_data,
      out_data,
      pos_data,
      channel_num,
      {row_lod_cpu.get(), static_cast<int>(row_lod.size()), row_lod_xpu},
      {col_lod_cpu.get(), static_cast<int>(col_lod.size()), col_lod_xpu},
      {topks.data(), static_cast<int>(k_num), topks_xpu});

  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(sequence_topk_avg_pooling,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SequenceTopkAvgPoolingCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("ROW", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("COLUMN", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("pos", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
