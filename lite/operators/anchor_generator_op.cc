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

#include "lite/operators/anchor_generator_op.h"
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool AnchorGeneratorOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.Input);
  CHECK_OR_FALSE(param_.Anchors);
  CHECK_OR_FALSE(param_.Variances);

  auto input_dims = param_.Input->dims();
  CHECK_OR_FALSE(input_dims.size() == 4);
  return true;
}

bool AnchorGeneratorOpLite::InferShapeImpl() const {
  auto input_dims = param_.Input->dims();
  size_t num_anchors = param_.aspect_ratios.size() * param_.anchor_sizes.size();
  std::vector<int64_t> output_shape(
      {input_dims[2], input_dims[3], static_cast<int64_t>(num_anchors), 4});
  param_.Anchors->Resize(output_shape);
  param_.Variances->Resize(output_shape);
  return true;
}

bool AnchorGeneratorOpLite::AttachImpl(const cpp::OpDesc &op_desc,
                                       lite::Scope *scope) {
  auto input_name = op_desc.Input("Input").front();
  auto anchor_name = op_desc.Output("Anchors").front();
  auto variances_name = op_desc.Output("Variances").front();

  param_.Input = scope->FindVar(input_name)->GetMutable<lite::Tensor>();
  param_.Anchors = scope->FindVar(anchor_name)->GetMutable<lite::Tensor>();
  param_.Variances = scope->FindVar(variances_name)->GetMutable<lite::Tensor>();
  param_.anchor_sizes = op_desc.GetAttr<std::vector<float>>("anchor_sizes");
  param_.aspect_ratios = op_desc.GetAttr<std::vector<float>>("aspect_ratios");
  param_.stride = op_desc.GetAttr<std::vector<float>>("stride");
  if (op_desc.HasAttr("variances")) {
    param_.variances = op_desc.GetAttr<std::vector<float>>("variances");
  }
  if (op_desc.HasAttr("offset")) {
    param_.offset = op_desc.GetAttr<float>("offset");
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(anchor_generator,
                 paddle::lite::operators::AnchorGeneratorOpLite);
