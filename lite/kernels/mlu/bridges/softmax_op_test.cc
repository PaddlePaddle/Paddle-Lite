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

#include "lite/operators/softmax_op.h"
#include <gtest/gtest.h>
#include "lite/core/op_registry.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

template <typename dtype>
void softmax_ref(const std::shared_ptr<operators::SoftmaxOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  auto x_data = x->data<dtype>();
  auto out_data = out->mutable_data<dtype>();
  DDim x_dims = x->dims();

  auto x_rank = x_dims.size();
  int axis = op_info->GetAttr<int>("axis");
  if (axis < 0) {
    axis += x_rank;
  }
  int axis_size = x_dims[axis];
  int outer_num = x_dims.Slice(0, axis).production();
  int inner_num = x_dims.Slice(axis + 1, x_rank).production();
  int compute_size = outer_num * inner_num;
  for (int i = 0; i < compute_size; i++) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int start = idx_outer * inner_num + idx_inner;
    int offset;

    offset = start;
    dtype max_data = std::numeric_limits<dtype>::lowest();
    for (int j = 0; j < axis_size; j++) {
      max_data = x_data[offset] > max_data ? x_data[offset] : max_data;
      offset += inner_num;
    }

    offset = start;
    dtype sum_data = (dtype)0;
    for (int j = 0; j < axis_size; j++) {
      out_data[offset] = exp(x_data[offset] - max_data);
      sum_data += out_data[offset];
      offset += inner_num;
    }

    offset = start;
    for (int j = 0; j < axis_size; j++) {
      out_data[offset] /= sum_data;
      offset += inner_num;
    }
  }
}

void test_softmax(const std::vector<int64_t>& input_shape, int axis) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name = "x";
  std::string out_var_name = "out";
  std::string out_ref_var_name = "out_ref";
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  x->Resize(input_shape);

  // initialize input&output data
  FillTensor<float>(x);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("softmax");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("axis", axis);

  // create and convert op to MLU model, then run it on MLU
  auto op = CreateOp<operators::SoftmaxOp>(opdesc, &scope);
  // execute reference implementation and save to output tensor
  softmax_ref<float>(op);
  out_ref->CopyDataFrom(*out);

  int bs = x->dims()[0];
  int ic = x->dims()[1];
  int ih = x->dims()[2];
  int iw = x->dims()[3];
  Tensor input_trans;
  input_trans.Resize({bs, ic, ih, iw});
  transpose(x->mutable_data<float>(),
            input_trans.mutable_data<float>(),
            {bs, ic, ih, iw},
            {0, 2, 3, 1});

  x->CopyDataFrom(input_trans);

  LaunchOp(op, {x_var_name}, {out_var_name});

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();
  Tensor output_trans;
  output_trans.Resize({bs, ic, ih, iw});
  transpose(out_data,
            output_trans.mutable_data<float>(),
            {bs, ih, iw, ic},
            {0, 3, 1, 2});
  out_data = output_trans.mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-2);
  }
}

TEST(MLUBridges, softmax) {
  // test_softmax({1, 4}, -1);
  // // Bug exists in HiAI DDK when the number of items > 16500
  // test_softmax({1, 16500}, -1);
  // test_softmax({1, 4}, 0);
  // test_softmax({1, 4}, 1);
  // test_softmax({3, 4}, -1);
  // test_softmax({3, 4}, 0);
  // test_softmax({3, 4}, 1);
  // test_softmax({1, 4, 7}, -1);
  // test_softmax({1, 4, 7}, 0);
  // // Bug exists in HiAI DDK when axis is 1 and iw > 1
  // // test_softmax({1, 4, 7}, 1);
  // test_softmax({1, 4, 1}, 1);
  // test_softmax({1, 4, 7}, 2);
  // test_softmax({3, 4, 7}, -1);
  // test_softmax({3, 4, 7}, 0);
  // test_softmax({3, 4, 1}, 1);
  // test_softmax({3, 4, 7}, 2);
  test_softmax({1, 4, 7, 9}, -1);
  test_softmax({1, 4, 7, 9}, 0);
  test_softmax({1, 4, 7, 9}, 1);
  // Bug exists in HiAI DDK when axis is 2 and iw > 1
  // test_softmax({1, 4, 7, 9}, 2);
  test_softmax({1, 4, 7, 1}, 2);
  test_softmax({1, 4, 7, 9}, 3);
  test_softmax({3, 4, 7, 9}, -1);
  test_softmax({3, 4, 7, 9}, 0);
  test_softmax({3, 4, 7, 9}, 1);
  test_softmax({3, 4, 7, 1}, 2);
  test_softmax({3, 4, 7, 9}, 3);
}

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(softmax, kMLU)
