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

  padding_idx_ = static_cast<int>(param.padding_idx);

  if (GetBoolFromEnv("XPU_PADDING_IDX", true)) {
    padding_idx_ = -1;
  }
  VLOG(3) << "model padding_idx: " << param.padding_idx
          << ", xpu padding_idx: " << padding_idx_;
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

  if (param.SeqLod && param.SeqLod->data<int>()) {
    auto batch_size = param.SeqLod->dims()[0] - 1;
    int pad_seq_len = param.PadSeqLen->data<int>()[0];

    auto* seq_lod = param.SeqLod->data<int>();
    idx_len = seq_lod[batch_size];

    for (size_t i = 0; i < emb_layer_num; ++i) {
      auto* idx_pad_ptr = param.Ids[i]->data<int64_t>();
      for (auto batch_idx = 0; batch_idx < batch_size; batch_idx++) {
        for (auto j = 0; j < seq_lod[batch_idx + 1] - seq_lod[batch_idx]; j++) {
          int_idx[i][seq_lod[batch_idx] + j] =
              static_cast<int>(idx_pad_ptr[batch_idx * pad_seq_len + j]);
        }
      }
      arg_ids_.push_back(
          xdnn::VectorParam<int>{int_idx[i].data(), idx_len, nullptr});
    }
    param.Out->Resize({1, idx_len, embed_dim});
    std::vector<int> out_lod0_int;
    out_lod0_int.insert(
        out_lod0_int.begin(), seq_lod, seq_lod + batch_size + 1);
    std::vector<uint64_t> out_lod0(out_lod0_int.begin(), out_lod0_int.end());
    paddle::lite::LoD out_lod;
    out_lod.push_back(out_lod0);
    param.Out->set_lod(out_lod);
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
      std::vector<int>(table_lens_cpu_.size(), padding_idx_));
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
    .BindInput("SeqLod",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindInput("PadSeqLen",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt32))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
