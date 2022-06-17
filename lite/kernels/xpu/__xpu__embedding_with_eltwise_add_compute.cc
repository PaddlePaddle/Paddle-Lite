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
  auto& param = this->template Param<param_t>();
  CHECK_GT(param.Tables.size(), 0);
  auto embed_dim = param.Tables[0]->dims()[1];
  for (auto* table : param.Tables) {
    auto& table_dims = table->dims();
    CHECK_EQ(table_dims.size(), 2); /* shape like [table_len, embed_dim] */
    CHECK_EQ(table_dims[1], embed_dim);
    table_lens_cpu_.push_back(table_dims[0]);
    arg_tables_.push_back(table->data<float>());
  }
}

void XPUEmbeddingWithEltwiseAddCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto& id_dims = param.Ids[0]->dims();
  int idx_len = id_dims[0] * id_dims[1];
  int emb_layer_num = param.Ids.size();
  int embed_dim = param.Tables[0]->dims()[1];
  std::vector<std::vector<int>> int_idx(emb_layer_num,
                                        std::vector<int>(idx_len, 0));
  std::vector<xdnn::VectorParam<int>> arg_ids_;

  if (param.Mask && param.Mask->data<float>()) {
    auto& mask_dims = param.Mask->dims();
    auto batch_size = mask_dims[0];
    auto pad_seq_len = mask_dims[1];
    param.PadSeqLen->mutable_data<int>()[0] = pad_seq_len;
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
      CHECK_GT(cur_batch_seq_len, 0);
    }
    auto* seq_lod_ptr = seq_lod->mutable_data<int>();
    memcpy(seq_lod_ptr, cpu_seq_lod.data(), cpu_seq_lod.size() * sizeof(int));
    idx_len = cpu_seq_lod.back();

    for (size_t i = 0; i < emb_layer_num; ++i) {
      auto* idx_pad_ptr = param.Ids[i]->data<int64_t>();
      for (auto batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (auto j = 0;
             j < cpu_seq_lod[batch_idx + 1] - cpu_seq_lod[batch_idx];
             j++) {
          int_idx[i][cpu_seq_lod[batch_idx] + j] =
              static_cast<int>(idx_pad_ptr[batch_idx * pad_seq_len + j]);
        }
      }
      arg_ids_.push_back(
          xdnn::VectorParam<int>{int_idx[i].data(), idx_len, nullptr});
    }
    param.Out->Resize({1, idx_len, embed_dim});
  } else {
    for (size_t i = 0; i < emb_layer_num; i++) {
      for (size_t j = 0; j < idx_len; j++) {
        int_idx[i][j] = static_cast<int>(param.Ids[i]->data<int64_t>()[j]);
      }
      arg_ids_.push_back(
          xdnn::VectorParam<int>{int_idx[i].data(), idx_len, nullptr});
    }
  }
  int r = xdnn::multi_embedding_fusion<float, float, int>(
      ctx.GetRawContext(),
      arg_tables_, /* tables */
      param.Out->mutable_data<float>(TARGET(kXPU)),
      arg_ids_,
      table_lens_cpu_,
      embed_dim,
      std::vector<float>(table_lens_cpu_.size(), 1.0f),
      std::vector<int>(table_lens_cpu_.size(),
                       static_cast<int>(param.padding_idx)));
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
    .BindInput("Ids", {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt64))})
    .BindInput("Tables", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("Mask", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("SeqLod",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("PadSeqLen",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .Finalize();
