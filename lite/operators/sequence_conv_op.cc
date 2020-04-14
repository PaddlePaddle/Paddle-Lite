// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/sequence_conv_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SequenceConvOp::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Filter);
  CHECK_OR_FALSE(param_.Out);

  // currently we only support the case that
  // the contextStride is equal to 1
  int context_length = param_.contextLength;
  int context_start = param_.contextStart;
  CHECK_EQ_OR_FALSE(param_.contextStride, 1UL);
  CHECK_GT_OR_FALSE(context_start, -context_length);
  CHECK_GE_OR_FALSE(0, context_start);

  const auto *filter = param_.Filter;
  auto lod = param_.X->lod();
  auto filter_dims = filter->dims();
  auto in_dims = param_.X->dims();
  CHECK_EQ_OR_FALSE(in_dims.size(), 2UL);
  CHECK_EQ_OR_FALSE(filter_dims.size(), 2UL);
  CHECK_EQ_OR_FALSE(lod.size(), 1UL);
  CHECK_EQ_OR_FALSE(filter_dims[0], context_length * in_dims[1]);
  CHECK_GE_OR_FALSE(in_dims[0], (static_cast<int64_t>(lod[0].size()) - 1));
  return true;
}

bool SequenceConvOp::InferShapeImpl() const {
  const auto *input = param_.X;
  const auto *filter = param_.Filter;
  auto in_dims = input->dims();
  auto filter_dims = filter->dims();
  auto out_dims = in_dims;
  out_dims[1] = filter_dims[1];
  param_.Out->Resize(out_dims);
  param_.Out->set_lod(param_.X->lod());
  return true;
}

bool SequenceConvOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  // required params
  param_.X = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("X").front())->Get<lite::Tensor>());
  param_.Filter = const_cast<lite::Tensor *>(
      &scope->FindVar(opdesc.Input("Filter").front())->Get<lite::Tensor>());
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  param_.contextStart = opdesc.GetAttr<int>("contextStart");
  param_.contextStride = opdesc.GetAttr<int>("contextStride");
  param_.contextLength = opdesc.GetAttr<int>("contextLength");

  // PaddingData is not supported for now
  std::vector<std::string> input_arg_names = opdesc.InputArgumentNames();
  if (std::find(input_arg_names.begin(),
                input_arg_names.end(),
                "PaddingData") != input_arg_names.end()) {
    auto padding_data_arguments = opdesc.Input("PaddingData");
    CHECK_EQ_OR_FALSE(padding_data_arguments.size(), 0);
  }

  // paddingTrainable == True is not supported for now.
  if (opdesc.HasAttr("paddingTrainable")) {
    CHECK_OR_FALSE(!opdesc.GetAttr<bool>("paddingTrainable"));
  }
  CHECK(param_.X);
  CHECK(param_.Filter);
  CHECK(param_.Out);
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sequence_conv, paddle::lite::operators::SequenceConvOp);
