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
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Index);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool GatherOp::InferShapeImpl() const {
  if (param_.Axis != nullptr || param_.axis != -1) {
    int axis_index = param_.axis;
    if (param_.Axis != nullptr) {
      if (param_.Axis->precision() == PRECISION(kInt32)) {
        auto *axis_data = param_.Axis->data<int32_t>();
        axis_index = axis_data[0];
      } else if (param_.Axis->precision() == PRECISION(kInt64)) {
        auto *axis_data = param_.Axis->data<int64_t>();
        axis_index = axis_data[0];
      } else {
        LOG(FATAL) << "Axis unsupport data type: "
                   << lite_api::PrecisionToStr(param_.Axis->precision());
      }
    }

    int index_size = param_.Index->numel();
    auto input_dim = param_.X->dims();

    int inner_dim_size = 1;
    int outer_dim_size = 1;
    std::vector<int64_t> out_dim_vec;
    for (int i = 0; i < axis_index; i++) {
      inner_dim_size *= input_dim[i];
      out_dim_vec.push_back(input_dim[i]);
    }
    out_dim_vec.push_back(index_size);
    for (int i = axis_index + 1; i < input_dim.size(); i++) {
      outer_dim_size *= input_dim[i];
      out_dim_vec.push_back(input_dim[i]);
    }
    param_.Out->Resize(out_dim_vec);
    return true;
  } else {
    auto index_dims = param_.Index->dims();
    CHECK(index_dims.size() == 1 ||
          (index_dims.size() == 2 && index_dims[1] == 1))
        << "index dims unmatch";
    int batch_size = index_dims[0];
    auto out_dims = param_.X->dims();
    out_dims[0] = batch_size;
    param_.Out->Resize(out_dims);
    return true;
  }
}

bool GatherOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X = scope->FindTensor(opdesc.Input("X").front());
  param_.Index = scope->FindTensor(opdesc.Input("Index").front());
  param_.Out = scope->FindMutableTensor(opdesc.Output("Out").front());
  if (opdesc.HasAttr("axis")) {
    param_.axis = opdesc.GetAttr<int>("axis");
  }
  if (opdesc.HasInput("Axis") && !opdesc.Input("Axis").empty()) {
    auto axis = opdesc.Input("Axis").front();
    param_.Axis = scope->FindTensor(axis);
    CHECK_EQ(param_.Axis->numel(), 1) << "value Axis size must be 1, but get"
                                      << param_.Axis->numel();
  }
  CHECK(param_.X) << "X is null";
  CHECK(param_.Index) << "index is null";
  CHECK(param_.Out) << "out is null";
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(gather, paddle::lite::operators::GatherOp);
