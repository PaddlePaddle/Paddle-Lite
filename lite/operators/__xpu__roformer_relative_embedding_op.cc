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

#include "lite/operators/__xpu__roformer_relative_embedding_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPURoformerRelativeEmbeddingOp::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.cos_embedding);
  CHECK_OR_FALSE(param_.sin_embedding);

  const auto input_dims = param_.input->dims();
  const auto cos_emb_dims = param_.cos_embedding->dims();
  const auto sin_emb_dims = param_.sin_embedding->dims();
  CHECK_EQ_OR_FALSE(input_dims.size(), 4UL);
  CHECK_EQ_OR_FALSE(cos_emb_dims.size(), 4UL);
  CHECK_EQ_OR_FALSE(sin_emb_dims.size(), 4UL);
  for (int i = 0; i < cos_emb_dims.size(); ++i) {
    CHECK_EQ(cos_emb_dims[i], sin_emb_dims[i]) << i << " dim embedding unmatch "
                                               << cos_emb_dims[i] << ", "
                                               << sin_emb_dims[i];
  }
  CHECK_EQ(input_dims[3], cos_emb_dims[3]) << input_dims[3] << ", "
                                           << cos_emb_dims[3];
  return true;
}

bool XPURoformerRelativeEmbeddingOp::InferShapeImpl() const {
  const auto& input_dims = param_.input->dims();
  param_.output->Resize(input_dims);
  // share LoD
  param_.output->set_lod(param_.input->lod());

  return true;
}

bool XPURoformerRelativeEmbeddingOp::AttachImpl(const cpp::OpDesc& op_desc,
                                                lite::Scope* scope) {
  param_.input =
      scope->FindVar(op_desc.Input("X").front())->GetMutable<Tensor>();
  param_.cos_embedding = scope->FindVar(op_desc.Input("CosEmbbeding").front())
                             ->GetMutable<Tensor>();
  param_.sin_embedding = scope->FindVar(op_desc.Input("SinEmbbeding").front())
                             ->GetMutable<Tensor>();
  param_.output =
      scope->FindVar(op_desc.Output("Out").front())->GetMutable<Tensor>();
  param_.max_pos_len = op_desc.GetAttr<int>("max_pos_len");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__roformer_relative_embedding,
                 paddle::lite::operators::XPURoformerRelativeEmbeddingOp);
