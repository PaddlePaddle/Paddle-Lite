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

#include "lite/operators/__xpu__sfa_head_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUSfaHeadOp::CheckShape() const {
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.output);
  CHECK_OR_FALSE(param_.op_type != "");

  const auto input_dims = param_.input->dims();
  if (param_.op_type == "meanstd" || param_.op_type == "moment") {
    CHECK_EQ_OR_FALSE(input_dims.size(), 3UL);
  }

  return true;
}

bool XPUSfaHeadOp::InferShapeImpl() const {
  const auto& input_dims = param_.input->dims();
  auto op_type = param_.op_type;

  // Set output dims
  std::vector<DDim::value_type> output_dims(2);
  output_dims[0] = input_dims[0];
  if (op_type == "meanstd") {
    output_dims[1] = 2 * input_dims[1];
  } else if (op_type == "moment") {
    output_dims[1] = 4 * input_dims[1];
  } else {
    LOG(FATAL) << "not supported vis op --> " << op_type;
  }
  param_.output->Resize(output_dims);

  // share LoD
  param_.output->set_lod(param_.input->lod());
  return true;
}

bool XPUSfaHeadOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  auto input = op_desc.Input("Input").front();
  auto output = op_desc.Output("Output").front();
  CHECK(scope->FindVar(input));
  CHECK(scope->FindVar(output));

  param_.input = scope->FindVar(input)->GetMutable<lite::Tensor>();
  param_.output = scope->FindVar(output)->GetMutable<lite::Tensor>();
  param_.op_type = op_desc.GetAttr<std::string>("op_type");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__sfa_head, paddle::lite::operators::XPUSfaHeadOp);
