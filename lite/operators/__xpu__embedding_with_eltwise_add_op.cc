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

#include "lite/operators/__xpu__embedding_with_eltwise_add_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite_metal {
namespace operators {

bool XPUEmbeddingWithEltwiseAddOp::CheckShape() const {
  CHECK_OR_FALSE(param_.Ids.size() == param_.Tables.size());

  auto& id_dims = param_.Ids[0]->dims();
  auto& table_dims = param_.Tables[0]->dims();

  int id_rank = id_dims.size();

  CHECK_EQ_OR_FALSE(table_dims.size(), 2);
  // id_dims must be [batch_size, seq_len] or [batch_size, seq_len, 1]
  CHECK(id_rank == 2 || id_rank == 3) << "unsupported id_rank: " << id_rank;

  if (param_.Mask != nullptr) {
    CHECK(param_.SeqLod != nullptr);
    CHECK(param_.PadSeqLen != nullptr);
  }
  return true;
}

bool XPUEmbeddingWithEltwiseAddOp::InferShapeImpl() const {
  auto& id_dims = param_.Ids[0]->dims();
  auto& table_dims = param_.Tables[0]->dims();

  std::vector<int64_t> out_shape{id_dims[0], id_dims[1], table_dims[1]};

  param_.Out->Resize(lite_metal::DDim(out_shape));
  param_.Out->set_lod(param_.Ids[0]->lod());
  if (param_.Mask != nullptr) {
    param_.PadSeqLen->Resize({1});
  }
  return true;
}

bool XPUEmbeddingWithEltwiseAddOp::AttachImpl(const cpp::OpDesc& op_desc,
                                              lite_metal::Scope* scope) {
  param_.Out = scope->FindVar(op_desc.Output("Output").front())
                   ->GetMutable<lite_metal::Tensor>();

  param_.Ids.clear();
  for (auto& name : op_desc.Input("Ids")) {
    auto t =
        const_cast<lite_metal::Tensor*>(&scope->FindVar(name)->Get<lite_metal::Tensor>());
    param_.Ids.push_back(t);
  }
  param_.Tables.clear();
  for (auto& name : op_desc.Input("Tables")) {
    auto t =
        const_cast<lite_metal::Tensor*>(&scope->FindVar(name)->Get<lite_metal::Tensor>());
    param_.Tables.push_back(t);
  }

  // optional params
  std::vector<std::string> input_arg_names = op_desc.InputArgumentNames();
  if (std::find(input_arg_names.begin(), input_arg_names.end(), "Mask") !=
      input_arg_names.end()) {
    auto arguments = op_desc.Input("Mask");
    if (arguments.size() > 0) {
      auto arg_var = scope->FindVar(arguments.front());
      if (arg_var != nullptr) {
        param_.Mask = &(arg_var->Get<lite_metal::Tensor>());
      }
    }
  }
  std::vector<std::string> output_arg_names = op_desc.OutputArgumentNames();
  if (std::find(output_arg_names.begin(), output_arg_names.end(), "SeqLod") !=
      output_arg_names.end()) {
    auto seqlod_name = op_desc.Output("SeqLod").front();
    param_.SeqLod = GetMutableVar<lite_metal::Tensor>(scope, seqlod_name);
  }
  if (std::find(output_arg_names.begin(),
                output_arg_names.end(),
                "PadSeqLen") != output_arg_names.end()) {
    auto padseqlen_name = op_desc.Output("PadSeqLen").front();
    param_.PadSeqLen = GetMutableVar<lite_metal::Tensor>(scope, padseqlen_name);
  }

  param_.padding_idx = op_desc.GetAttr<int64_t>("padding_idx");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__embedding_with_eltwise_add,
                 paddle::lite_metal::operators::XPUEmbeddingWithEltwiseAddOp);
