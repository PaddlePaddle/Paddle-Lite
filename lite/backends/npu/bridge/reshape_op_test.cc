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

#include "lite/operators/reshape_op.h"
#include <gtest/gtest.h>
#include <random>
#include "lite/backends/npu/bridge/registry.h"
#include "lite/backends/npu/bridge/test_helper.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

void reshape_ref(const std::shared_ptr<lite::OpLite> op) {
  auto scope = op->scope();
  auto op_info = op->op_info();
  auto op_type = op_info->Type();
  auto x = scope->FindVar(op_info->Input("X").front())->GetMutable<Tensor>();
  auto out =
      scope->FindVar(op_info->Output("Out").front())->GetMutable<Tensor>();
  auto x_dims = x->dims();
  auto shape = op_info->GetAttr<std::vector<int>>("shape");
  auto inplace = op_info->GetAttr<bool>("inplace");
  if (op_info->HasInput("Shape")) {
    auto actual_shape_var_names = op_info->Input("Shape");
    if (actual_shape_var_names.size() > 0) {
      auto actual_shape = scope->FindVar(actual_shape_var_names.front())
                              ->GetMutable<lite::Tensor>();
      auto actual_shape_dims = actual_shape->dims();
      auto* actual_shape_data = actual_shape->data<int>();
      shape =
          std::vector<int>(actual_shape_data,
                           actual_shape_data + actual_shape_dims.production());
    }
  }
  if (inplace) {
    out->ShareDataWith(*x);
  } else {
    out->CopyDataFrom(*x);
  }
  auto out_dims = operators::ValidateShape(shape, x_dims);
  out->Resize(out_dims);
}

void test_reshape(const std::vector<int64_t>& x_shape,
                  const std::vector<int>& shape,
                  const std::vector<int>& act_shape,
                  bool inplace,
                  bool reshape2) {
  // prepare input&output variables
  Scope scope;
  std::string x_var_name("x");
  std::string actual_shape_var_name("actual_shape");
  std::string out_var_name("out");
  std::string out_ref_var_name("out_ref");
  std::string xshape_var_name("xshape");
  std::string xshape_ref_var_name("xshape_ref");
  auto x = scope.Var(x_var_name)->GetMutable<Tensor>();
  auto actual_shape = scope.Var(actual_shape_var_name)->GetMutable<Tensor>();
  auto out = scope.Var(out_var_name)->GetMutable<Tensor>();
  auto out_ref = scope.Var(out_ref_var_name)->GetMutable<Tensor>();
  auto xshape = scope.Var(xshape_var_name)->GetMutable<Tensor>();
  auto xshape_ref = scope.Var(xshape_ref_var_name)->GetMutable<Tensor>();

  x->Resize(x_shape);

  // initialize input&output data
  FillTensor<float, int>(x);

  // initialize op desc
  cpp::OpDesc opdesc;
  opdesc.SetType(reshape2 ? "reshape2" : "reshape");
  opdesc.SetInput("X", {x_var_name});
  opdesc.SetOutput("Out", {out_var_name});
  opdesc.SetAttr("shape", shape);
  opdesc.SetAttr("inplace", inplace);
  if (!act_shape.empty()) {
    int64_t act_shape_size = act_shape.size();
    actual_shape->Resize({act_shape_size});
    memcpy(actual_shape->mutable_data<int>(),
           act_shape.data(),
           act_shape_size * sizeof(int));
    opdesc.SetInput("Shape", {actual_shape_var_name});
  }
  if (reshape2) {
    opdesc.SetOutput("XShape", {xshape_var_name});
  }

  // create op and execute reference implementation
  auto op = reshape2 ? CreateOp<operators::Reshape2Op>(opdesc, &scope)
                     : CreateOp<operators::ReshapeOp>(opdesc, &scope);
  reshape_ref(op);
  out_ref->CopyDataFrom(*out);
  if (reshape2) {
    xshape_ref->CopyDataFrom(*xshape);
  }

  // convert op to NPU model, then run it on NPU
  LauchOp(op,
          {x_var_name},
          {out_var_name});  // TODO(hong19860320) support XShape for reshape2

  // compare results
  auto out_dims = out->dims();
  auto out_ref_dims = out_ref->dims();
  CHECK_EQ(out_dims.size(), out_ref_dims.size());
  for (int i = 0; i < out_dims.size(); i++) {
    CHECK_EQ(out_dims[i], out_ref_dims[i]);
  }
  auto out_data = out->mutable_data<float>();
  auto out_ref_data = out_ref->mutable_data<float>();
  for (int i = 0; i < out->dims().production(); i++) {
    VLOG(5) << i;
    EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
  }
  // if (reshape2) {
  //   auto xshape_dims = xshape->dims();
  //   auto xshape_ref_dims = xshape_ref->dims();
  //   CHECK_EQ(xshape_dims.size(), xshape_ref_dims.size());
  //   for (size_t i = 0; i < xshape_dims.size(); i++) {
  //     CHECK_EQ(xshape_dims[i], xshape_ref_dims[i]);
  //   }
  // }
}

TEST(NPUBridges, reshape) {
#if 1
  std::map<std::vector<int64_t>, std::vector<std::vector<int>>> tests = {
      {{1, 2, 4, 6},
       {{},
        {-1},
        {48},
        {-1, 48},
        {1, 48},
        {0, 48},
        {48, -1},
        {48, 1},
        {-1, 24},
        {2, 24},
        {24, 0},
        {-1, 0, 3, 2},
        {4, 2, 3, 2},
        {0, -1, 3, 2},
        {1, 8, 3, 2}}}};
  for (auto& i : tests) {
    for (auto& shape : i.second) {
      if (shape.empty()) {
        continue;
      }
      for (auto& act_shape : i.second) {
        for (auto& inplace : {true, false}) {
          for (auto& reshape2 : {true, false}) {
            std::stringstream ss;
            ss << "x:{ ";
            for (auto s : i.first) {
              ss << s << " ";
            }
            ss << "} shape:{ ";
            for (auto s : shape) {
              ss << s << " ";
            }
            ss << "} act_shape:{ ";
            for (auto s : act_shape) {
              ss << s << " ";
            }
            VLOG(3) << ss.str() << "} inplace:" << inplace
                    << " reshape2:" << reshape2;
            test_reshape(i.first, shape, act_shape, inplace, reshape2);
          }
        }
      }
    }
  }
#else
  test_reshape({2, 4, 6}, {-1, 0, 4, 3}, {}, true, true);
  test_reshape({1, 232, 14, 14}, {-1, 2, 116, 14, 14}, {}, true, true);
#endif
}

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle

USE_LITE_OP(reshape);
USE_NPU_BRIDGE(reshape);

USE_LITE_OP(reshape2);
USE_NPU_BRIDGE(reshape2);
