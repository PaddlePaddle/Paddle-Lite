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

#include "lite/operators/write_to_array_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool WriteToArrayOp::CheckShape() const { return true; }

bool WriteToArrayOp::InferShape() const {
  auto in_dims = param_.X[0]->dims();
  for (auto out : param_.Out) {
    out->Resize(in_dims);
  }
  return true;
}

bool WriteToArrayOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  auto inputs = opdesc.Input("X");
  LOG(INFO) << "inputs_size" << inputs.size();
  for (auto in : inputs) {
    LOG(INFO) << in;
    param_.X.push_back(scope->FindVar(in)->GetMutable<lite::Tensor>());
  }
  LOG(INFO) << "inputs have been prepared";
  auto outputs = opdesc.Output("Out");
  LOG(INFO) << "outputs_size" << outputs.size();
  for (auto out : outputs) {
    param_.Out.push_back(scope->FindVar(out)->GetMutable<lite::Tensor>());
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(write_to_array, paddle::lite::operators::WriteToArrayOp);
