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

#include "lite/operators/eye_op.h"

#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool EyeOpLite::CheckShape() const {
  CHECK(param_.out);
  return true;
}

bool EyeOpLite::InferShapeImpl() const {
  switch (param_.dtype) {
    case 0:
      param_.out->set_precision(PRECISION(kBool));
      break;
    case 1:
      param_.out->set_precision(PRECISION(kInt16));
      break;
    case 2:
      param_.out->set_precision(PRECISION(kInt32));
      break;
    case 3:
      param_.out->set_precision(PRECISION(kInt64));
      break;
#ifdef ENABLE_ARM_FP16
    case 4:
      param_.out->set_precision(PRECISION(kFP16));
      break;
#endif
    case 5:
      param_.out->set_precision(PRECISION(kFloat));
      break;
    case 20:
      param_.out->set_precision(PRECISION(kUInt8));
      break;
    case 21:
      param_.out->set_precision(PRECISION(kInt8));
      break;
    default:
      param_.out->set_precision(PRECISION(kFloat));
      break;
  }

  param_.out->Resize(std::vector<int64_t>{param_.num_rows, param_.num_columns});
  return true;
}

bool EyeOpLite::AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto out_name = opdesc.Output("Out").front();
  param_.out = GetMutableVar<lite::Tensor>(scope, out_name);

  if (opdesc.GetAttrType("num_rows") == OpAttrType::INT)
    param_.num_rows = static_cast<int64_t>(opdesc.GetAttr<int>("num_rows"));
  else
    param_.num_rows = opdesc.GetAttr<int64_t>("num_rows");
  if (opdesc.GetAttrType("num_columns") == OpAttrType::INT)
    param_.num_columns =
        static_cast<int64_t>(opdesc.GetAttr<int>("num_columns"));
  else
    param_.num_columns = opdesc.GetAttr<int64_t>("num_columns");
  param_.dtype = opdesc.GetAttr<int>("dtype");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(eye, paddle::lite::operators::EyeOpLite);
