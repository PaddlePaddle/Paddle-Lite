// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/xpu/__xpu__roformer_relative_embedding_compute.h"
#include <vector>
#include "lite/backends/xpu/xpu_header_sitter.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

void RoformerRelativeEmbeddingCompute::Run() {
  auto& param = this->template Param<param_t>();
  auto& ctx = this->ctx_->template As<XPUContext>();
  auto input_dim = param.input->dims();
  CHECK_EQ(input_dim.size(), 4);
  int batch = input_dim[0];
  int head_num = param.input->dims()[1];
  int seqlen = param.input->dims()[2];
  int head_dim = param.input->dims()[3];
  CHECK_LE(seqlen, param.max_pos_len);
  std::vector<int> lod;
  lod.resize(batch + 1);
  for (int i = 0; i < batch + 1; i++) {
    lod[i] = i * seqlen;
  }
  int r =
      xdnn::rope<float>(ctx.GetRawContext(),
                        param.input->data<float>(),
                        param.output->mutable_data<float>(TARGET(kXPU)),
                        param.cos_embedding->data<float>(),
                        param.sin_embedding->data<float>(),
                        batch,
                        head_num,
                        head_dim,
                        head_num * head_dim,
                        lod,
                        param.max_pos_len,
                        false,  // no vsl
                        true);  // transpose to [n, seql, head_num, head_dim]
  CHECK_EQ(r, 0) << "call RoformerRelativeEmbeddingCompute failed";
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    __xpu__roformer_relative_embedding,
    kXPU,
    kFloat,
    kNCHW,
    paddle::lite::kernels::xpu::RoformerRelativeEmbeddingCompute,
    def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("CosEmbbeding", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindInput("SinEmbbeding", {LiteType::GetTensorTy(TARGET(kXPU))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kXPU))})
    .Finalize();
