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

#include "lite/operators/batch_norm_op.h"
#include <gtest/gtest.h>
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

template <typename dtype>
void batch_norm_ref(const std::shared_ptr<operators::BatchNormOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto y = scope->FindVar(op_info->Output("Y").front())->GetMutable<Tensor>();
  auto bias =
      scope->FindVar(op_info->Input("Bias").front())->GetMutable<Tensor>();
  auto scale =
      scope->FindVar(op_info->Input("Scale").front())->GetMutable<Tensor>();
  auto mean =
      scope->FindVar(op_info->Input("Mean").front())->GetMutable<Tensor>();
  auto variance =
      scope->FindVar(op_info->Input("Variance").front())->GetMutable<Tensor>();

  auto x_data = x->data<dtype>();
  auto y_data = y->mutable_data<dtype>();
  auto scale_data = scale->mutable_data<dtype>();
  auto bias_data = bias->mutable_data<dtype>();
  auto mean_data = mean->mutable_data<dtype>();
  auto variance_data = variance->mutable_data<dtype>();
  DDim x_dims = x->dims();

  float epsilon = op_info->GetAttr<float>("epsilon");
  float momentum = op_info->GetAttr<float>("momentum");
  auto data_layout = op_info->GetAttr<std::string>("data_layout");

  bool global_stats = op_info->GetAttr<bool>("use_global_stats");
  if (global_stats) {
    int64_t outer_size = 0;
    int64_t channel_size = 0;
    int64_t inner_size = 0;
    if (data_layout == "NCHW") {
      outer_size = x_dims[0];
      channel_size = x_dims[1];
      inner_size = x_dims.Slice(2, x_dims.size()).production();
    } else {
      LOG(FATAL) << "Unknown storage order: " << data_layout;
    }
    auto x_ptr = x_data;
    auto y_ptr = y_data;
    for (int o = 0; o < outer_size; o++) {
      for (int c = 0; c < channel_size; c++) {
        for (int i = 0; i < inner_size; i++) {
          dtype norm_x =
              (*x_ptr - mean_data[c]) / std::sqrt(variance_data[c] + epsilon);
          *y_ptr = norm_x * scale_data[c] + bias_data[c];
          x_ptr++;
          y_ptr++;
        }
      }
    }
  }
}

void test_batch_norm(
    int bs, int ic, int ih, int iw, float epsilon, float momentum) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name = "x";
  std::string out_var_name = "out";
  std::string out_ref_var_name = "out_ref";
  std::string scale_var_name = "scale";
  std::string bias_var_name = "bias";
  std::string mean_var_name = "mean";
  std::string variance_var_name = "variance";
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* scale = scope.Var(scale_var_name)->GetMutable<Tensor>();
  auto* bias = scope.Var(bias_var_name)->GetMutable<Tensor>();
  auto* mean = scope.Var(mean_var_name)->GetMutable<Tensor>();
  auto* variance = scope.Var(variance_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize({bs, ic, ih, iw});
  scale->Resize({ic});
  bias->Resize({ic});
  mean->Resize({ic});
  variance->Resize({ic});

  // initialize input&output data
  FillTensor<float, int>(x);
  FillTensor<float, int>(scale);
  FillTensor<float, int>(bias);
  FillTensor<float, int>(mean);
  // variance > 0
  FillTensor<float, int>(variance, 1.f, 5.f);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("batch_norm");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetInput("Scale", {scale_var_name});
  opdesc.SetInput("Bias", {bias_var_name});
  opdesc.SetInput("Mean", {mean_var_name});
  opdesc.SetInput("Variance", {variance_var_name});
  opdesc.SetOutput("Y", {out_var_name});
  opdesc.SetAttr("is_test", 1);
  opdesc.SetAttr("use_global_stats", true);
  opdesc.SetAttr("epsilon", epsilon);
  opdesc.SetAttr("momentum", momentum);
  opdesc.SetAttr("data_layout", std::string("NCHW"));

  // create and convert op to NPU model, then run it on NPU
  auto op = CreateOp<operators::BatchNormOp>(opdesc, &scope);
  LauchOp(op, {x_var_name}, {out_var_name});
  out_ref->CopyDataFrom(*out);

  // execute reference implementation and save to output tensor
  batch_norm_ref<float>(op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }
}

TEST(NPUBridges, batch_norm) {
  for (auto bs : {1, 4, 7}) {
    for (auto ic : {1, 4, 7}) {
      for (auto ih : {1, 4, 7}) {
        for (auto iw : {1, 4, 7}) {
          for (auto epsilon : {1e-4f, 1e-5f}) {
            for (auto momentum : {0.9f, 0.99f}) {
              test_batch_norm(bs, ic, ih, iw, epsilon, momentum);
            }
          }
        }
      }
    }
  }
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(batch_norm);
USE_NPU_BRIDGE(batch_norm);
