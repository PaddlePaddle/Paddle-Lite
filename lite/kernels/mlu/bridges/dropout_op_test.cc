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

#include "lite/operators/dropout_op.h"
#include <gtest/gtest.h>
#include <random>
#include "lite/core/op_registry.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

void dropout_ref(const std::shared_ptr<operators::DropoutOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  auto dropout_implementation =
      op_info->GetAttr<std::string>("dropout_implementation");
  auto dropout_prob = op_info->GetAttr<float>("dropout_prob");
  float alpha = 1.0f - dropout_prob;
  if (dropout_implementation == "upscale_in_train") {
    alpha = 1.;
  }
  float beta = 0.;

  auto x_data = x->data<float>();
  auto out_data = out->mutable_data<float>();
  DDim x_dims = x->dims();
  DDim out_dims = out->dims();
  CHECK_EQ(x_dims.production(), out_dims.production());
  for (int i = 0; i < out_dims.production(); i++) {
    out_data[i] = x_data[i] * alpha + beta;
  }
}

void test_dropout(int bs,
                  int ic,
                  int ih,
                  int iw,
                  std::string dropout_implementation,
                  float dropout_prob,
                  float bias) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name("x");
  std::string out_var_name("out");
  std::string mask_var_name("mask");
  std::string out_ref_var_name("out_ref");
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* mask = scope.Var(mask_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize({bs, ic, ih, iw});

  // initialize input&output data
  FillTensor<float, int>(x);

  // initialize op desc
  bool is_test = true;
  bool fix_seed = false;
  int seed = 0;
  cpp::OpDesc opdesc;
  opdesc.SetType("dropout");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetOutput("Mask", {mask_var_name});
  opdesc.SetAttr("is_test", is_test);
  opdesc.SetAttr("fix_seed", fix_seed);
  opdesc.SetAttr("seed", seed);
  opdesc.SetAttr("dropout_implementation", dropout_implementation);
  opdesc.SetAttr("dropout_prob", dropout_prob);
  VLOG(6) << "mask: " << mask->dims()[0] << std::endl;
  // create and convert op to MLU model, then run it on MLU
  auto op = CreateOp<operators::DropoutOp>(opdesc, &scope);
  dropout_ref(op);
  out_ref->CopyDataFrom(*out);

  Tensor input_trans;
  input_trans.Resize({bs, ic, ih, iw});
  transpose(x->mutable_data<float>(),
            input_trans.mutable_data<float>(),
            {bs, ic, ih, iw},
            {0, 2, 3, 1});
  auto os = out->dims();
  out->Resize({static_cast<int>(os[0]),
               static_cast<int>(os[2]),
               static_cast<int>(os[3]),
               static_cast<int>(os[1])});
  x->CopyDataFrom(input_trans);
  x->Resize({bs, ih, iw, ic});

  LaunchOp(op, {x_var_name}, {out_var_name});

  // execute reference implementation and save to output tensor('out')

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  Tensor output_trans;
  output_trans.Resize(os);
  transpose(out_data,
            output_trans.mutable_data<float>(),
            {static_cast<int>(os[0]),
             static_cast<int>(os[2]),
             static_cast<int>(os[3]),
             static_cast<int>(os[1])},
            {0, 3, 1, 2});
  out_data = output_trans.mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    VLOG(5) << i;
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }
}

TEST(MLUBridges, dropout) {
  for (auto bs : {1, 3}) {
    for (auto ic : {1, 3}) {
      for (auto ih : {3, 4}) {
        for (auto iw : {4, 3}) {
          for (auto dropout_implementation :
               {"downgrade_in_infer", "upscale_in_train"}) {
            for (auto dropout_prob : {0.f, 1.0f}) {
              VLOG(3) << "bs: " << bs << " ic: " << ic << " ih: " << ih
                      << " iw: " << iw
                      << " dropout_implementation: " << dropout_implementation
                      << " dropout_prob: " << dropout_prob;
              test_dropout(
                  bs, ic, ih, iw, dropout_implementation, dropout_prob, 0.);
            }
          }
        }
      }
    }
  }
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(dropout, kMLU);
