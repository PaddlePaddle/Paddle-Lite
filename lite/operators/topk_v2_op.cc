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

#include "lite/operators/topk_v2_op.h"
#include "lite/core/op_registry.h"
namespace paddle {
namespace lite {
namespace operators {

bool TopkV2Op::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  CHECK_OR_FALSE(param_.Indices);
  return true;
}

bool TopkV2Op::InferShapeImpl() const {
  auto out_dims = param_.X->dims();
  int dim_size = out_dims.size();
  auto axis_valid =
      ((param_.axis >= (-1 * dim_size)) && (param_.axis < dim_size));
  CHECK_EQ(axis_valid, true) << "the axis of topk_v2 must be ["
                             << (-1 * dim_size) << ", " << dim_size
                             << "but you set axis is" << param_.axis;
  if (param_.axis < 0) {
    param_.axis += dim_size;
  }
  int k = -1;
  if (param_.k_is_tensor) {
    k = param_.KTensor->data<int>()[0];
  } else {
    k = param_.K;
  }
  CHECK_GE(out_dims[param_.axis], k) << "input of topk_v2 op must have >=" << k
                                     << " columns in axis of "
                                     << out_dims[param_.axis];
  out_dims[param_.axis] = k;
  auto out = param_.Out;
  out->Resize(out_dims);
  out->set_lod(param_.X->lod());

  auto indices = param_.Indices;
  indices->Resize(out_dims);
  indices->set_lod(param_.X->lod());

  return true;
}

bool TopkV2Op::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  param_.X = scope->FindTensor(x);

  auto output0 = op_desc.Output("Out").front();
  auto output1 = op_desc.Output("Indices").front();
  param_.Out = scope->FindMutableTensor(output0);
  param_.Indices = scope->FindMutableTensor(output1);
  if (op_desc.HasInput("K") && op_desc.Input("K").size() > 0) {
    param_.KTensor = scope->FindTensor(op_desc.Input("K").front());
    param_.k_is_tensor = true;
  } else {
    param_.K = op_desc.GetAttr<int>("k");
    param_.k_is_tensor = false;
  }
  param_.axis = op_desc.GetAttr<int>("axis");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(top_k_v2, paddle::lite::operators::TopkV2Op);
