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

#include "lite/operators/gather_op.h"
#include <algorithm>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool GatherOp::CheckShape() const {
  LOG(INFO) << "check shape";
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Index);
  CHECK_OR_FALSE(param_.Out);

  auto index_dims = param_.Index->dims();
  if (index_dims.size() == 2) {
    CHECK_EQ(index_dims[1], 1)
        << "The last dim of index should be 1 when it is 2D, but we get "
        << index_dims[1];
  } else {
    CHECK_EQ(index_dims.size(), 1)
        << "The index should be 1D, when it is not 2D, but we get "
        << index_dims.size();
  }

  return true;
}

bool GatherOp::InferShapeImpl() const {
  LOG(INFO) << "InferShapeImpl";

  auto axis = param_.axis;
  auto input_dim = param_.X->dims();
  auto index_dims = param_.Index->dims();
  if (param_.Axis != nullptr || axis == 0) {
    // if has Axis, we can not obtain correct shape of output
    int batch_size = index_dims[0];
    DDim output_dims(input_dim);
    output_dims[0] = batch_size;
    param_.Out->Resize(output_dims);
  } else {
    int index_size = index_dims[0];
    std::vector<int64_t> out_dim_vec;
    for (int i = 0; i < axis; i++) {
      out_dim_vec.push_back(input_dim[i]);
    }
    out_dim_vec.push_back(index_size);
    for (int i = axis + 1; i < input_dim.size(); i++) {
      out_dim_vec.push_back(input_dim[i]);
    }
    param_.Out->Resize(DDim(out_dim_vec));
  }

  return true;
}

bool GatherOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  LOG(INFO) << "AttachImpl";
  param_.X = scope->FindTensor(opdesc.Input("X").front());
  param_.Index = scope->FindTensor(opdesc.Input("Index").front());
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());
  if (opdesc.HasInput("Axis") && !opdesc.Input("Axis").empty()) {
    LOG(INFO) << "AXIS exist";
    auto axis = opdesc.Input("Axis").front();
    LOG(INFO) << "xxx";
    param_.Axis = scope->FindTensor(axis);
  }
  param_.axis = opdesc.GetAttr<int>("axis");
  CHECK(param_.X) << "X is null";
  CHECK(param_.Index) << "index is null";
  CHECK(param_.Out) << "out is null";

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(gather, paddle::lite::operators::GatherOp);
