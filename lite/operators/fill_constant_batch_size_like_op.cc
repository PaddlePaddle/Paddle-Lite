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

#include "lite/operators/fill_constant_batch_size_like_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool FillConstantBatchSizeLikeOp::CheckShape() const {
  CHECK(param_.out);
  CHECK(param_.input);
  CHECK_GT(param_.shape.size(), 0u);
  CHECK_GE(param_.input_dim_idx, 0);
  CHECK_GE(param_.output_dim_idx, 0);
  return true;
}

bool FillConstantBatchSizeLikeOp::InferShapeImpl() const {
  std::vector<int64_t> output_dim{param_.shape.begin(), param_.shape.end()};
  if (param_.input_dim_idx == 0 && !param_.input->lod().empty()) {
    output_dim[param_.output_dim_idx] = param_.input->lod().back().size() - 1;
  } else {
    output_dim[param_.output_dim_idx] =
        param_.input->dims()[param_.input_dim_idx];
  }
  param_.out->Resize(output_dim);
  return true;
}

bool FillConstantBatchSizeLikeOp::AttachImpl(const cpp::OpDesc& opdesc,
                                             lite::Scope* scope) {
  auto out_name = opdesc.Output("Out").front();
  auto input_name = opdesc.Input("Input").front();

  param_.out = GetMutableVar<lite::Tensor>(scope, out_name);
  param_.input = GetMutableVar<lite::Tensor>(scope, input_name);
  param_.dtype = opdesc.GetAttr<int>("dtype");
  param_.shape = opdesc.GetAttr<std::vector<int>>("shape");
  if (opdesc.HasAttr("value")) {
    param_.value = opdesc.GetAttr<float>("value");
  }
  if (opdesc.HasAttr("input_dim_idx")) {
    param_.input_dim_idx = opdesc.GetAttr<int>("input_dim_idx");
  }
  if (opdesc.HasAttr("output_dim_idx")) {
    param_.output_dim_idx = opdesc.GetAttr<int>("output_dim_idx");
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(fill_constant_batch_size_like,
                 paddle::lite::operators::FillConstantBatchSizeLikeOp);
