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

#include "lite/operators/__xpu__bigru_op.h"
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool XPUBiGRUOp::CheckShape() const {
  CHECK_OR_FALSE(param_.input)
  CHECK_OR_FALSE(param_.fw_mul_w)
  CHECK_OR_FALSE(param_.fw_gru_w)
  CHECK_OR_FALSE(param_.bw_mul_w)
  CHECK_OR_FALSE(param_.bw_gru_w)
  CHECK_OR_FALSE(param_.fw_output)
  CHECK_OR_FALSE(param_.bw_output)

  CHECK_GT_OR_FALSE(param_.input->dims().size(),
                    static_cast<size_t>(param_.fw_mul_x_num_col_dims))
  CHECK_GT_OR_FALSE(param_.input->dims().size(),
                    static_cast<size_t>(param_.bw_mul_x_num_col_dims))
  CHECK_GT_OR_FALSE(param_.fw_mul_w->dims().size(),
                    static_cast<size_t>(param_.fw_mul_y_num_col_dims))
  CHECK_GT_OR_FALSE(param_.bw_mul_w->dims().size(),
                    static_cast<size_t>(param_.bw_mul_y_num_col_dims))

  int fw_gru_frame_size = param_.fw_gru_w->dims()[0];
  CHECK_EQ_OR_FALSE(param_.fw_mul_w->dims()[0], param_.input->dims()[1])
  CHECK_EQ_OR_FALSE(param_.fw_mul_w->dims()[1], fw_gru_frame_size * 3)
  CHECK_EQ_OR_FALSE(param_.fw_gru_w->dims()[0], fw_gru_frame_size)
  CHECK_EQ_OR_FALSE(param_.fw_gru_w->dims()[1], fw_gru_frame_size * 3)
  int bw_gru_frame_size = param_.bw_gru_w->dims()[0];
  CHECK_EQ_OR_FALSE(param_.bw_mul_w->dims()[0], param_.input->dims()[1])
  CHECK_EQ_OR_FALSE(param_.bw_mul_w->dims()[1], bw_gru_frame_size * 3)
  CHECK_EQ_OR_FALSE(param_.bw_gru_w->dims()[0], bw_gru_frame_size)
  CHECK_EQ_OR_FALSE(param_.bw_gru_w->dims()[1], bw_gru_frame_size * 3)
  CHECK_EQ_OR_FALSE(fw_gru_frame_size, bw_gru_frame_size);

  if (param_.fw_mul_b) {
    auto bias_dims = param_.fw_mul_b->dims();
    int bias_width = bias_dims[0];
    CHECK_EQ_OR_FALSE(bias_width, fw_gru_frame_size * 3)
  }
  if (param_.bw_mul_b) {
    auto bias_dims = param_.bw_mul_b->dims();
    int bias_width = bias_dims[0];
    CHECK_EQ_OR_FALSE(bias_width, bw_gru_frame_size * 3)
  }
  if (param_.fw_gru_b) {
    auto bias_dims = param_.fw_gru_b->dims();
    int bias_height = bias_dims[0];
    int bias_width = bias_dims[1];
    CHECK_EQ_OR_FALSE(bias_height, 1)
    CHECK_EQ_OR_FALSE(bias_width, fw_gru_frame_size * 3)
  }
  if (param_.bw_gru_b) {
    auto bias_dims = param_.bw_gru_b->dims();
    int bias_height = bias_dims[0];
    int bias_width = bias_dims[1];
    CHECK_EQ_OR_FALSE(bias_height, 1)
    CHECK_EQ_OR_FALSE(bias_width, bw_gru_frame_size * 3)
  }

  return true;
}

bool XPUBiGRUOp::InferShapeImpl() const {
  int batch_size = param_.input->dims()[0];
  int fw_gru_frame_size = param_.fw_gru_w->dims()[0];
  param_.fw_output->Resize(lite::DDim({batch_size, fw_gru_frame_size}));
  *(param_.fw_output->mutable_lod()) = param_.input->lod();
  int bw_gru_frame_size = param_.bw_gru_w->dims()[0];
  param_.bw_output->Resize(lite::DDim({batch_size, bw_gru_frame_size}));
  *(param_.bw_output->mutable_lod()) = param_.input->lod();
  return true;
}

bool XPUBiGRUOp::AttachImpl(const cpp::OpDesc& op_desc, lite::Scope* scope) {
  bool has_mul_b = op_desc.GetAttr<bool>("has_mul_b");
  bool has_gru_b = op_desc.GetAttr<bool>("has_gru_b");

  param_.input =
      scope->FindVar(op_desc.Input("Input").front())->GetMutable<Tensor>();
  param_.fw_mul_w = scope->FindVar(op_desc.Input("ForwardMulWeight").front())
                        ->GetMutable<Tensor>();
  param_.bw_mul_w = scope->FindVar(op_desc.Input("BackwardMulWeight").front())
                        ->GetMutable<Tensor>();
  if (has_mul_b) {
    param_.fw_mul_b = scope->FindVar(op_desc.Input("ForwardMulBias").front())
                          ->GetMutable<Tensor>();
  }
  if (has_mul_b) {
    param_.bw_mul_b = scope->FindVar(op_desc.Input("BackwardMulBias").front())
                          ->GetMutable<Tensor>();
  }
  param_.fw_gru_w = scope->FindVar(op_desc.Input("ForwardGRUWeight").front())
                        ->GetMutable<Tensor>();
  param_.bw_gru_w = scope->FindVar(op_desc.Input("BackwardGRUWeight").front())
                        ->GetMutable<Tensor>();
  if (has_gru_b) {
    param_.fw_gru_b = scope->FindVar(op_desc.Input("ForwardGRUBias").front())
                          ->GetMutable<Tensor>();
  }
  if (has_gru_b) {
    param_.bw_gru_b = scope->FindVar(op_desc.Input("BackwardGRUBias").front())
                          ->GetMutable<Tensor>();
  }
  param_.fw_output = scope->FindVar(op_desc.Output("ForwardOutput").front())
                         ->GetMutable<Tensor>();
  param_.bw_output = scope->FindVar(op_desc.Output("BackwardOutput").front())
                         ->GetMutable<Tensor>();

  param_.fw_mul_x_num_col_dims = op_desc.GetAttr<int>("fw_mul_x_num_col_dims");
  param_.fw_mul_y_num_col_dims = op_desc.GetAttr<int>("fw_mul_y_num_col_dims");
  param_.bw_mul_x_num_col_dims = op_desc.GetAttr<int>("bw_mul_x_num_col_dims");
  param_.bw_mul_y_num_col_dims = op_desc.GetAttr<int>("bw_mul_y_num_col_dims");
  param_.fw_gru_gate_activation =
      op_desc.GetAttr<std::string>("fw_gru_gate_activation");
  param_.bw_gru_gate_activation =
      op_desc.GetAttr<std::string>("bw_gru_gate_activation");
  param_.fw_gru_activation = op_desc.GetAttr<std::string>("fw_gru_activation");
  param_.bw_gru_activation = op_desc.GetAttr<std::string>("bw_gru_activation");
  if (op_desc.HasAttr("fw_gru_origin_mode")) {
    param_.fw_gru_origin_mode = op_desc.GetAttr<bool>("fw_gru_origin_mode");
  }
  if (op_desc.HasAttr("bw_gru_origin_mode")) {
    param_.bw_gru_origin_mode = op_desc.GetAttr<bool>("bw_gru_origin_mode");
  }

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(__xpu__bigru, paddle::lite::operators::XPUBiGRUOp);
