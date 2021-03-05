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

#include "lite/operators/meshgrid_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool MeshgridOpLite::CheckShape() const {
  int x_size = param_.X.size();
  int out_size = param_.Out.size();
  CHECK_GE(x_size, 1) << "Input(X) should not be empty.";
  CHECK_GE(out_size, 1) << "Output(Out) should not be empty.";
  CHECK_LE(x_size, 6) << "The rank of Input(X) must not be greater than 6.";
  return true;
}

bool MeshgridOpLite::InferShapeImpl() const {
  int inputs_num = param_.X.size();
  int outputs_num = param_.Out.size();
  std::vector<int64_t> out_shape(inputs_num);
  for (size_t i = 0; i < inputs_num; ++i) {
    out_shape[i] = param_.X[i]->dims()[0];
  }
  for (size_t i = 0; i < outputs_num; ++i) {
    param_.Out[i]->Resize(out_shape);
  }
  return true;
}

bool MeshgridOpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto input_list = opdesc.Input("X");
  param_.X.clear();
  for (auto var : input_list) {
    param_.X.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  auto output_list = opdesc.Output("Out");
  param_.Out.clear();
  for (auto var : output_list) {
    param_.Out.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(meshgrid, paddle::lite::operators::MeshgridOpLite);
