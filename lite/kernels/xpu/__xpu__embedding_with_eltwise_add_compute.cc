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
namespace lite_metal {
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
  table_lens_guard_ = TargetWrapperXPU::MallocScratchPad(lens_size);
  XPU_CALL(xpu_memcpy(table_lens_guard_->addr_,
                      &table_lens_cpu_[0],
                      lens_size,
                      XPU_HOST_TO_DEVICE));
  idx_guard_ = TargetWrapperXPU::MallocScratchPad(32768 * sizeof(int64_t));
}

void XPUEmbeddingWithEltwiseAddCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->As<XPUContext>();
  auto& id_dims = param.Ids[0]->dims();
  int idx_len = id_dims[0] * id_dims[1];
  int emb_layer_num = param.Ids.size();
  auto& table_dims = param.Tables[0]->dims();
  int embed_dim = table_dims[1];
  for (size_t i = 0; i < param.Tables.size(); ++i) {
    arg_tables_[i] = param.Tables[i]->data<float>();
  }
  if (param.Mask && param.Mask->data<float>()) {
    auto& mask_dims = param.Mask->dims();
    auto batch_size = mask_dims[0];
    auto pad_seq_len = mask_dims[1];
    param.PadSeqLen->mutable_data<int>()[0] = pad_seq_len;
    CHECK_EQ(batch_size, id_dims[0]);
    CHECK_EQ(idx_len, param.Mask->numel());
    auto* seq_lod = param.SeqLod;
    seq_lod->Resize({batch_size + 1});
    std::vector<int> cpu_seq_lod{0};
    auto* mask_ptr = param.Mask->data<float>();
    for (auto batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      int cur_batch_seq_len = 0;
      for (auto seq_idx = 0; seq_idx < pad_seq_len; seq_idx++) {
        if (mask_ptr[batch_idx * pad_seq_len + seq_idx] > 1e-7) {
          cur_batch_seq_len += 1;
        } else {
          break;
        }
      }
      cpu_seq_lod.push_back(cpu_seq_lod.back() + cur_batch_seq_len);
    }
    auto* seq_lod_ptr = seq_lod->mutable_data<int>();
    memcpy(seq_lod_ptr, cpu_seq_lod.data(), cpu_seq_lod.size() * sizeof(int));
    idx_len = cpu_seq_lod.back();

    idx_guard_->Reserve(emb_layer_num * idx_len * sizeof(int64_t));
    int64_t* idx_xpu_ptr = static_cast<int64_t*>(idx_guard_->addr_);
    std::vector<std::vector<int64_t>> idx_remove_pad(
        emb_layer_num, std::vector<int64_t>(idx_len, 0));
    for (size_t i = 0; i < emb_layer_num; ++i) {
      auto* idx_pad_ptr = param.Ids[i]->data<int64_t>();
      for (auto batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        memcpy(&idx_remove_pad[i][cpu_seq_lod[batch_idx]],
               idx_pad_ptr + batch_idx * pad_seq_len,
               sizeof(int64_t) *
                   (cpu_seq_lod[batch_idx + 1] - cpu_seq_lod[batch_idx]));
      }
      XPU_CALL(xpu_memcpy(idx_xpu_ptr + i * idx_len,
                          &idx_remove_pad[i][0],
                          sizeof(int64_t) * idx_len,
                          XPU_HOST_TO_DEVICE));
      arg_ids_[i] = idx_xpu_ptr + i * idx_len;
    }
    param.Out->Resize({1, idx_len, embed_dim});
  } else {
    idx_guard_->Reserve(emb_layer_num * idx_len * sizeof(int64_t));
    int64_t* idx_xpu_ptr = static_cast<int64_t*>(idx_guard_->addr_);
    for (size_t i = 0; i < emb_layer_num; ++i) {
      CHECK_EQ(idx_len, param.Ids[i]->numel());
      XPU_CALL(xpu_memcpy(idx_xpu_ptr + idx_len * i,
                          param.Ids[i]->data<int64_t>(),
                          sizeof(int64_t) * idx_len,
                          XPU_HOST_TO_DEVICE));
      arg_ids_[i] = idx_xpu_ptr + idx_len * i;
    }
  }
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
    paddle::lite_metal::kernels::xpu::XPUEmbeddingWithEltwiseAddCompute,
    def)
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Tables", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("SeqLod",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("PadSeqLen",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
