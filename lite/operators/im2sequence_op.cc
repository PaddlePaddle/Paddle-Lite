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

#include "lite/operators/im2sequence_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {
inline int Im2SeqOutputSize(
    int input_size, int filter_size, int padding_0, int padding_1, int stride) {
  const int output_size =
      (input_size + padding_0 + padding_1 - filter_size) / stride + 1;
  return output_size;
}

bool Im2SequenceOp::CheckShape() const { return true; }
bool Im2SequenceOp::InferShapeImpl() const {
  CHECK_OR_FALSE(param_.Out);
  // TODO(Superjomn) Enable data sharing.
  auto input_dims = param_.X->dims();
  int img_num = input_dims[0];
  int img_channels = input_dims[1];
  int img_height = input_dims[2];
  int img_width = input_dims[3];
  auto kernels = param_.kernels;
  auto paddings = param_.paddings;
  auto strides = param_.strides;
  DDimLite out_dims(
      std::vector<int64_t>({1, img_channels * kernels[0] * kernels[1]}));

  int output_height = Im2SeqOutputSize(
      img_height, kernels[0], paddings[0], paddings[1], strides[0]);
  int output_width = Im2SeqOutputSize(
      img_width, kernels[1], paddings[2], paddings[3], strides[1]);

  out_dims[0] = img_num * output_height * output_width;
  param_.Out->Resize(out_dims);

  // share lod
  // param_.Out->set_lod(param_.X->lod());
  return true;
}

bool Im2SequenceOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.X =
      scope->FindVar(opdesc.Input("X").front())->GetMutable<lite::Tensor>();
  if (opdesc.HasInput("Y") && opdesc.Input("Y").size()) {
    param_.Y =
        scope->FindVar(opdesc.Input("Y").front())->GetMutable<lite::Tensor>();
  }
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  CHECK(param_.Out);
  param_.strides = opdesc.GetAttr<std::vector<int>>("strides");

  // same with paddle: pad[top, left, down, right]-->[top, down, left, right]
  std::vector<int> tmp = opdesc.GetAttr<std::vector<int>>("paddings");
  param_.paddings[0] = tmp[0];
  param_.paddings[1] = tmp[2];
  param_.paddings[2] = tmp[1];
  param_.paddings[3] = tmp[3];

  param_.kernels = opdesc.GetAttr<std::vector<int>>("kernels");
  if (opdesc.HasAttr("out_stride")) {
    param_.out_strides = opdesc.GetAttr<std::vector<int>>("out_stride");
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(im2sequence, paddle::lite::operators::Im2SequenceOp);
