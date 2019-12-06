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

#include "lite/operators/split_op.h"
#include <gtest/gtest.h>
#include "lite/core/op_registry.h"
#include "lite/kernels/npu/bridges/registry.h"
#include "lite/kernels/npu/bridges/test_helper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace npu {
namespace bridges {

template <typename dtype>
void split_ref(const std::shared_ptr<operators::SplitOp> op) {
  Scope* scope = op->scope();
  const OpInfo* op_info = op->op_info();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  int num = op_info->GetAttr<int>("num");
  int axis = op_info->GetAttr<int>("axis");
  std::vector<int> sections = op_info->GetAttr<std::vector<int>>("sections");
  std::vector<lite::Tensor*> output_vec;
  auto output = op_info->Output("Out");
  for (auto out_var : output) {
    output_vec.push_back(scope->Var(out_var)->GetMutable<Tensor>());
  }
  auto in_dims = x->dims();
  auto rank = in_dims.size();
  int outs_number = output_vec.size();
  std::vector<lite::DDimLite> outs_dims;
  outs_dims.reserve(outs_number);
  if (axis < 0) {
    axis += rank;
  }
  if (num > 0) {
    int out_axis_dim = in_dims[axis] / num;
    for (int i = 0; i < outs_number; ++i) {
      auto dim = in_dims;
      dim[axis] = out_axis_dim;
      outs_dims.push_back(dim);
    }
  } else if (sections.size() > 0) {
    for (size_t i = 0; i < outs_number; ++i) {
      auto dim = in_dims;
      dim[axis] = sections[i];
      outs_dims.push_back(dim);
    }
  }
  for (int j = 0; j < outs_dims.size(); ++j) {
    output_vec[j]->Resize(outs_dims[j]);
  }

  const dtype* din = x->mutable_data<const dtype>();
  std::vector<int> in_strides(in_dims.size());
  in_strides[in_dims.size() - 1] = in_dims[in_dims.size() - 1];
  for (int i = in_dims.size() - 2; i >= 0; --i) {
    in_strides[i] = in_strides[i + 1] * in_dims[i];
  }

  int input_offset = 0;
  for (auto out : output_vec) {
    auto out_dim = out->dims();
    std::vector<int> out_strides(out_dim.size());
    out_strides[out_dim.size() - 1] = out_dim[out_dim.size() - 1];
    for (int i = out_dim.size() - 2; i >= 0; --i) {
      out_strides[i] = out_strides[i + 1] * out_dim[i];
    }

    dtype* out_data = out->mutable_data<dtype>();
    int before = out_strides[0] / out_strides[axis];
    int in_after = in_strides[axis];
    int out_after = out_strides[axis];

    for (int i = 0; i < before; ++i) {
      std::memcpy(out_data + i * out_after,
                  din + input_offset + i * in_after,
                  sizeof(dtype) * out_after);
    }
    input_offset += out_strides[axis];
  }
}

void test_split(int bs,
                int ic,
                int ih,
                int iw,
                int axis,
                int num,
                std::vector<int> sections) {
  const auto& bridges = lite::kernels::npu::bridges::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  CHECK(bridges.HasType("split"));
  // prepare input&output variables
  std::string x_var_name = "x";
  std::string out_var_name_1 = "out_1";
  std::string out_var_name_2 = "out_2";
  std::string out_ref_var_name_1 = "out_ref_1";
  std::string out_ref_var_name_2 = "out_ref_2";

  Scope scope;
  auto* x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto* out_1 = scope.Var(out_var_name_1)->GetMutable<Tensor>();
  auto* out_2 = scope.Var(out_var_name_2)->GetMutable<Tensor>();
  auto* out_ref_1 = scope.Var(out_ref_var_name_1)->GetMutable<Tensor>();
  auto* out_ref_2 = scope.Var(out_ref_var_name_2)->GetMutable<Tensor>();
  x->Resize({bs, ic, ih, iw});
  // initialize input&output data
  FillTensor<float>(x);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType("split");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name_1, out_var_name_2});
  opdesc.SetAttr("axis", axis);
  opdesc.SetAttr("sections", sections);
  opdesc.SetAttr("num", num);
  // create and convert op to NPU model, then run it on NPU
  auto op = CreateOp<operators::SplitOp>(opdesc, &scope);
  LauchOp(op, {x_var_name}, {out_var_name_1, out_var_name_2});
  out_ref_1->CopyDataFrom(*out_1);
  out_ref_2->CopyDataFrom(*out_2);
  // execute reference implementation and save to output tensor
  split_ref<float>(op);

  // compare results
  auto* out_data_1 = out_1->mutable_data<float>();
  auto* out_data_2 = out_2->mutable_data<float>();
  auto* out_ref_data_1 = out_ref_1->mutable_data<float>();
  auto* out_ref_data_2 = out_ref_2->mutable_data<float>();
  for (int i = 0; i < out_1->dims().production(); i++) {
    VLOG(5) << i;
    EXPECT_NEAR(out_data_1[i], out_ref_data_1[i], 5e-4);
  }
  for (int i = 0; i < out_2->dims().production(); i++) {
    VLOG(5) << i;
    EXPECT_NEAR(out_data_2[i], out_ref_data_2[i], 5e-4);
  }
}

TEST(NPUBridges, split) {
  test_split(4, 2, 3, 1, 0, 2, {});
  test_split(4, 2, 3, 1, 0, 0, {3, 1});
  test_split(4, 6, 3, 1, 1, 2, {});
  test_split(4, 6, 3, 1, 1, 0, {2, 4});
  test_split(4, 2, 2, 1, 2, 2, {});
  test_split(4, 2, 6, 1, 2, 0, {3, 3});
  test_split(4, 2, 3, 4, 3, 2, {});
  test_split(4, 2, 3, 6, 3, 0, {5, 1});
}

}  // namespace bridges
}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(split);
USE_NPU_BRIDGE(split);
