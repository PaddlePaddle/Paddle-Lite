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

#include "lite/operators/graph_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool GraphOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.output);
  return true;
}

bool GraphOpLite::InferShape() const { return CheckShape(); /* enrich me */ }

bool GraphOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("Input").front();
  auto out = op_desc.Output("Output").front();

  CHECK(scope->FindVar(x));
  CHECK(scope->FindVar(out));
  param_.input = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.output = scope->FindVar(out)->GetMutable<lite::Tensor>();
  param_.graph_name = op_desc.GetAttr<std::string>("graph_name");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(graph_op, paddle::lite::operators::GraphOpLite);
