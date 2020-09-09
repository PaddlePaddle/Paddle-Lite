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

#include "lite/kernels/xpu/__xpu__embedding_with_eltwise_add_compute.h"
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void XPUEmbeddingWithEltwiseAddCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();

  arg_ids_.reserve(param.Ids.size());
  arg_tables_.reserve(param.Tables.size());
  for (auto* table : param.Tables) {
    auto& table_dims = table->dims();
    CHECK_EQ(table_dims.size(), 2); /* shape like [table_len, embed_dim] */
    table_lens_cpu_.push_back(table_dims[0]);
  }

  size_t lens_size = table_lens_cpu_.size() * sizeof(int);
  table_lens_guard_ =
      TargetWrapperXPU::MallocScratchPad(lens_size, false /* use_l3 */);
  XPU_CALL(xpu_memcpy(table_lens_guard_->addr_,
                      &table_lens_cpu_[0],
                      lens_size,
                      XPU_HOST_TO_DEVICE));
}

void XPUEmbeddingWithEltwiseAddCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();

  for (size_t i = 0; i < param.Ids.size(); ++i) {
    arg_ids_[i] = param.Ids[i]->data<int64_t>();
  }
  for (size_t i = 0; i < param.Tables.size(); ++i) {
    arg_tables_[i] = param.Tables[i]->data<float>();
  }

  auto& id_dims = param.Ids[0]->dims();
  auto& table_dims = param.Tables[0]->dims();
  int idx_len = id_dims[0] * id_dims[1];
  int embed_dim = table_dims[1];
  int emb_layer_num = param.Ids.size();
  int r = xdnn::embedding_with_ewadd<float, int64_t, false, false>(
      ctx.GetRawContext(),                         /* context */
      embed_dim,                                   /* embed_dim */
      idx_len,                                     /* idx_len */
      emb_layer_num,                               /* emb_layer_num */
      param.padding_idx,                           /* padding_idx */
      &arg_tables_[0],                             /* tables */
      &arg_ids_[0],                                /* indices */
      static_cast<int*>(table_lens_guard_->addr_), /* table_lens */
      nullptr,                                     /* scale_after_emb */
      nullptr,                                     /* scale_after_ewadd */
      param.Out->mutable_data<float>(TARGET(kXPU)) /* top */);
  CHECK_EQ(r, 0);
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    __xpu__embedding_with_eltwise_add,
    kXPU,
    kFloat,
    kNCHW,
    paddle::lite::kernels::xpu::XPUEmbeddingWithEltwiseAddCompute,
    def)
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kXPU), PRECISION(kInt64))})
    .BindInput("Tables", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
