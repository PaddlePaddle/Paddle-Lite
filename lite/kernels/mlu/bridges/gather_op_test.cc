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

#include "lite/operators/gather_op.h"
#include <gtest/gtest.h>
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/subgraph_bridge_registry.h"
#include "lite/kernels/mlu/bridges/test_helper.h"
#include "lite/kernels/mlu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

template <typename dtype>
void gather_ref(const std::shared_ptr<operators::GatherOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto index =
      scope->FindVar(op_info->Input("Index").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();

  auto x_dims = x->dims();
  auto index_dims = index->dims();
  CHECK(index_dims.size() == 1 ||
        (index_dims.size() == 2 && index_dims[1] == 1));

  int batch_size = index_dims[0];
  DDim out_dims = x_dims;
  out_dims[0] = batch_size;
  out->Resize(out_dims);

  auto x_data = x->data<float>();
  auto index_data = index->data<int>();
  auto out_data = out->mutable_data<float>();

  auto slice_num = x_dims[0];
  auto slice_size = x_dims.Slice(1, x_dims.size()).production();
  for (int i = 0; i < batch_size; i++) {
    auto index = index_data[i];
    CHECK_LT(index, slice_num) << "index <= slice_num";
    CHECK_GE(index, 0) << "index > 0";
    memcpy(out_data + i * slice_size,
           x_data + index * slice_size,
           slice_size * sizeof(float));
  }
}

void test_gather() {
  // prepare input&output variables
  std::string x_var_name = "x";
  std::string out_var_name = "out";
  std::string out_ref_var_name = "out_ref";
  std::string index_var_name = "index";

  Scope scope;
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto* out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  auto* index = scope.Var(index_var_name)->GetMutable<Tensor>();

  x->Resize({5, 4, 3, 2});
  index->Resize({2});
  // initialize input&output data
  FillTensor<float>(x);
  FillTensor<int>(index, 1, 3);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("gather");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetInput("Index", {index_var_name});
  opdesc.SetOutput("Out", {out_var_name});

  auto op = CreateOp<operators::GatherOp>(opdesc, &scope);
  gather_ref<float>(op);
  out_ref->CopyDataFrom(*out);

  Tensor input;
  input.Resize({5, 4, 3, 2});
  transpose<float>(x->mutable_data<float>(),
                   input.mutable_data<float>(),
                   {static_cast<int>(5),
                    static_cast<int>(4),
                    static_cast<int>(3),
                    static_cast<int>(2)},
                   {0, 2, 3, 1});
  x->CopyDataFrom(input);
  LaunchOp(op, {x_var_name, index_var_name}, {out_var_name});

  // compare results
  auto* out_data = out->mutable_data<float>();
  auto* out_ref_data = out_ref->mutable_data<float>();

  Tensor output;
  output.Resize(out->dims());
  transpose<float>(out_data,
                   output.mutable_data<float>(),
                   {static_cast<int>(out->dims()[0]),
                    static_cast<int>(out->dims()[2]),
                    static_cast<int>(out->dims()[3]),
                    static_cast<int>(out->dims()[1])},
                   {0, 3, 1, 2});
  out_data = output.mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    VLOG(5) << i;
    EXPECT_NEAR(out_data[i], out_ref_data[i], 5e-4);
  }
}

TEST(MLUBridges, gather) { test_gather(); }

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

USE_SUBGRAPH_BRIDGE(gather, kMLU);
