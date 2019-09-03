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

#include "lite/operators/shuffle_channel_op.h"
#include <gtest/gtest.h>
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

void shuffle_channel_ref(
    const std::shared_ptr<operators::ShuffleChannelOpLite> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  auto x_data = x->mutable_data<float>();
  auto out_data = out->mutable_data<float>();
  int group = op_info->GetAttr<int>("group");
  auto x_dims = x->dims();

  int n_size = x_dims.production() / x_dims[0];
  int c_size = n_size / x_dims[1];
  for (int n = 0; n < x_dims[0]; n++) {
    int g_num = x_dims[1] / group;
    auto tmp_out_data = out_data;
    for (int g = 0; g < g_num; g++) {
      auto tmp_x_data = x_data + g * c_size;
      for (int i = 0; i < group; i++) {
        std::memcpy(tmp_out_data,
                    tmp_x_data + i * g_num * c_size,
                    c_size * sizeof(float));
        tmp_out_data += c_size;
      }
    }
    x_data += n_size;
    out_data += n_size;
  }
}

void test_shuffle_channel(int bs, int ic, int ih, int iw, int group) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name = "x";
  std::string out_var_name = "out";
  std::string out_ref_var_name = "out_ref";
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize({bs, ic, ih, iw});

  // initialize input&output data
  FillTensor<float>(x);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("shuffle_channel");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("group", group);

  // create and convert op to NPU model, then run it on NPU
  auto op = CreateOp<operators::ShuffleChannelOpLite>(opdesc, &scope);
  LauchOp(op, {x_var_name}, {out_var_name});
  out_ref->CopyDataFrom(*out);

  // execute reference implementation and save to output tensor
  shuffle_channel_ref(op);

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }
}

TEST(NPUBridges, softmax) {
  for (auto bs : {1, 4}) {
    for (auto ic : {1, 24, 35}) {
      for (auto ih : {1, 4}) {
        for (auto iw : {1, 4}) {
          for (auto group : {1, 3, 7, 24, 35}) {
            if (ic % group != 0) continue;
            test_shuffle_channel(bs, ic, ih, iw, group);
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

USE_LITE_OP(shuffle_channel);
USE_NPU_BRIDGE(shuffle_channel);
