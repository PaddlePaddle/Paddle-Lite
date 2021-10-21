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

#include "lite/operators/tensor_array_to_tensor_op.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool TensorArrayToTensorOpLite::CheckShape() const {
  CHECK_GE_OR_FALSE(param_.X->size(), 1UL);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool TensorArrayToTensorOpLite::InferShapeImpl() const {
  std::vector<Tensor *> inputs;
  for (int i = 0; i < param_.X->size(); i++) {
    inputs.push_back(&(*param_.X)[i]);
  }
  const size_t n = inputs.size();
  int axis = param_.axis;
  bool use_stack = param_.use_stack;
  if (use_stack) {
    auto input_dims = inputs[0]->dims();
    int rank = input_dims.size();
    if (axis < 0) axis += (rank + 1);
    auto vec = input_dims.Vectorize();
    vec.insert(vec.begin() + axis, inputs.size());
    param_.Out->Resize(vec);
  } else {
    auto out_dims = inputs[0]->dims();
    size_t in_zero_dims_size = out_dims.size();
    for (size_t i = 1; i < n; i++) {
      const auto &input_dims_i = inputs[i]->dims();
      for (size_t j = 0; j < in_zero_dims_size; j++) {
        if (j == static_cast<size_t>(axis)) {
          out_dims[axis] += input_dims_i[j];
        } else {
          CHECK_EQ_OR_FALSE(out_dims[j], input_dims_i[j]);
        }
      }
    }
    if (out_dims[axis] < 0) {
      out_dims[axis] = -1;
    }
    param_.Out->Resize(out_dims);
    auto out_lod = param_.Out->mutable_lod();
    *out_lod = inputs[0]->lod();
  }
  auto index_dim = param_.OutIndex->dims();
  if (index_dim.empty()) {
    std::vector<int64_t> index;
    index.push_back(n);
    index_dim.ConstructFrom(index);
  } else {
    index_dim[0] = n;
  }
  param_.OutIndex->Resize(index_dim);

  return true;
}

bool TensorArrayToTensorOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                           lite::Scope *scope) {
  auto out = op_desc.Output("Out").front();
  auto outIndex = op_desc.Output("OutIndex").front();

  auto in = op_desc.Input("X").front();
  param_.X = scope->FindVar(in)->GetMutable<std::vector<Tensor>>();

  CHECK(scope->FindVar(out));
  param_.Out = scope->FindVar(out)->GetMutable<lite::Tensor>();
  param_.OutIndex = scope->FindVar(outIndex)->GetMutable<lite::Tensor>();
  param_.axis = op_desc.GetAttr<int>("axis");
  param_.use_stack = op_desc.GetAttr<bool>("use_stack");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(tensor_array_to_tensor,
                 paddle::lite::operators::TensorArrayToTensorOpLite);
