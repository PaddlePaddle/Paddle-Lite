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

#include "lite/operators/rnn_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool RnnOp::CheckShape() const {
  CHECK_OR_FALSE(param_.Input);
  return true;
}

bool RnnOp::InferShapeImpl() const {
  auto in_dims = param_.Input->dims();
  CHECK_EQ(in_dims.size(), 3) << "input dims should be 3";
  int batch = in_dims[1];
  int seq = in_dims[0];
  int hidden_size = param_.hidden_size;
  bool is_bidirec = param_.is_bidirec;
  int out_hidden_size = is_bidirec ? 2 * hidden_size : hidden_size;
  DDimLite out_dims(std::vector<int64_t>{seq, batch, out_hidden_size});
  param_.Out->Resize(out_dims);
  param_.State.resize(param_.PreState.size());
  for (int i = 0; i < param_.PreState.size(); i++) {
    DDimLite state_dims = param_.PreState[i]->dims();
    param_.State[i]->Resize(state_dims);
  }
  return true;
}

bool RnnOp::AttachImpl(const cpp::OpDesc &opdesc, lite::Scope *scope) {
  param_.Input =
      scope->FindVar(opdesc.Input("Input").front())->GetMutable<lite::Tensor>();

  auto PreState = opdesc.Input("PreState");
  param_.PreState.clear();
  for (auto var : PreState) {
    param_.PreState.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  auto WeightList = opdesc.Input("WeightList");
  param_.WeightList.clear();
  for (auto var : WeightList) {
    param_.WeightList.push_back(
        scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  if (opdesc.HasInput("SequenceLength") &&
      !opdesc.Input("SequenceLength").empty()) {
    param_.SequenceLength =
        scope->FindTensor(opdesc.Input("SequenceLength").front());
  }
  param_.DropoutState = scope->FindVar(opdesc.Output("DropoutState").front())
                            ->GetMutable<lite::Tensor>();
  param_.Reserve = scope->FindVar(opdesc.Output("Reserve").front())
                       ->GetMutable<lite::Tensor>();
  param_.Out =
      scope->FindVar(opdesc.Output("Out").front())->GetMutable<lite::Tensor>();
  auto State = opdesc.Output("State");
  param_.State.clear();
  for (auto var : State) {
    param_.State.push_back(scope->FindVar(var)->GetMutable<lite::Tensor>());
  }
  param_.dropout_prob = opdesc.GetAttr<float>("dropout_prob");
  param_.is_bidirec = opdesc.GetAttr<bool>("is_bidirec");
  param_.input_size = opdesc.GetAttr<int>("input_size");
  param_.hidden_size = opdesc.GetAttr<int>("hidden_size");
  param_.num_layers = opdesc.GetAttr<int>("num_layers");
  param_.mode = opdesc.GetAttr<std::string>("mode");
  param_.is_test = opdesc.GetAttr<bool>("is_test");
  param_.seed = opdesc.GetAttr<int>("seed");

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(rnn, paddle::lite::operators::RnnOp);
