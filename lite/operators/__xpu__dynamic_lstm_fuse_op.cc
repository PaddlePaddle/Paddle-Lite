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

#include "lite/operators/__xpu__dynamic_lstm_fuse_op.h"
#include <memory>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUDynamicLstmOp::CheckShape() const {
  // check for xpu_fc
  CHECK_OR_FALSE(param_.input);
  CHECK_OR_FALSE(param_.weight_0);

  const auto input_dims = param_.input->dims();
  const auto w0_dims = param_.weight_0->dims();
  CHECK_EQ_OR_FALSE(w0_dims.size(), 2UL);

  int64_t w0_dims_1 = w0_dims[1];
  const auto bias_dims = param_.bias_0->dims();
  if (bias_dims.size() == 2) {
    CHECK_EQ_OR_FALSE(bias_dims[0], 1);
    CHECK_EQ_OR_FALSE(bias_dims[1], w0_dims_1);
  } else if (bias_dims.size() == 1) {
    CHECK_EQ_OR_FALSE(bias_dims[0], w0_dims_1);
  }

  // check for lstm
  CHECK_OR_FALSE(param_.weight_1);
  CHECK_OR_FALSE(param_.bias_1);
  if (param_.h0) {
    CHECK(param_.c0) << "lstm must has H0 and C0 in the same time";
    auto h_dims = param_.h0->dims();
    auto c_dims = param_.c0->dims();
    CHECK_EQ(h_dims, c_dims) << "H0 and C0 dims must be same";
  }

  std::vector<DDim::value_type> mid_dims(2);
  mid_dims[0] = input_dims[0];
  mid_dims[1] = w0_dims_1;

  int frame_size = mid_dims[1] / 4;
  auto w1_dims = param_.weight_1->dims();
  CHECK_EQ(w1_dims.size(), 2) << "weight dims should be 2";
  CHECK_EQ(w1_dims[0], frame_size) << "weight first dims should be "
                                   << frame_size;
  CHECK_EQ(w1_dims[1], 4 * frame_size) << "weight dims should be 4 * "
                                       << frame_size;
  auto b_dims = param_.bias_1->dims();
  CHECK_EQ(b_dims.size(), 2) << "Bias dims should be 2";
  CHECK_EQ(b_dims[0], 1) << "Bias first dims should be 1";
  CHECK_EQ(b_dims[1], 4 * frame_size) << "Bias second dim must be 4 * "
                                      << frame_size;
  return true;
}

bool XPUDynamicLstmOp::InferShapeImpl() const {
  const auto& input_dims = param_.input->dims();
  const auto& w0_dims = param_.weight_0->dims();
  int64_t w0_dims_1 = w0_dims[1];

  // Set output dims
  std::vector<DDim::value_type> mid_dims(2);
  mid_dims[0] = input_dims[0];
  mid_dims[1] = w0_dims_1;

  int frame_size = mid_dims[1] / 4;
  auto w1_dims = param_.weight_1->dims();
  DDimLite out_dims(std::vector<int64_t>{mid_dims[0], frame_size});

  param_.hidden->Resize(out_dims);

  auto hidden_lod = param_.hidden->mutable_lod();
  *hidden_lod = param_.input->lod();
  return true;
}

bool XPUDynamicLstmOp::AttachImpl(const cpp::OpDesc& op_desc,
                                  lite::Scope* scope) {
  CHECK(scope->FindVar(op_desc.Input("Input").front()));
  CHECK(scope->FindVar(op_desc.Input("Weight_0").front()));
  CHECK(scope->FindVar(op_desc.Input("Weight_1").front()));
  CHECK(scope->FindVar(op_desc.Input("Bias_0").front()));
  CHECK(scope->FindVar(op_desc.Input("Bias_1").front()));
  CHECK(scope->FindVar(op_desc.Output("Hidden").front()));
  param_.has_h0 = op_desc.GetAttr<bool>("has_h0");
  param_.is_reverse = op_desc.GetAttr<bool>("is_reverse");
  param_.input =
      scope->FindVar(op_desc.Input("Input").front())->GetMutable<Tensor>();
  param_.weight_0 =
      scope->FindVar(op_desc.Input("Weight_0").front())->GetMutable<Tensor>();
  param_.weight_1 =
      scope->FindVar(op_desc.Input("Weight_1").front())->GetMutable<Tensor>();
  param_.bias_0 =
      scope->FindVar(op_desc.Input("Bias_0").front())->GetMutable<Tensor>();
  param_.bias_1 =
      scope->FindVar(op_desc.Input("Bias_1").front())->GetMutable<Tensor>();
  param_.hidden =
      scope->FindVar(op_desc.Output("Hidden").front())->GetMutable<Tensor>();

  // optional params
  if (param_.has_h0) {
    param_.h0 =
        scope->FindVar(op_desc.Input("H0").front())->GetMutable<Tensor>();
    param_.c0 =
        scope->FindVar(op_desc.Input("C0").front())->GetMutable<Tensor>();
  }
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__dynamic_lstm_fuse_op,
                 paddle::lite::operators::XPUDynamicLstmOp);
