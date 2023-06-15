// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <memory>
#include <string>
#include "lite/backends/xpu/math.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class MatmulScaleSoftmaxV1fuser : public FuseBase {
 public:
  MatmulScaleSoftmaxV1fuser(bool with_cast, bool with_inner_scale)
      : with_cast_(with_cast), with_inner_scale_(with_inner_scale) {}

  void BuildPattern() override {
    auto* mat_q =
        VarNode("mat_q")->assert_is_op_input("matmul_v2", "X")->AsInput();
    auto* mat_k =
        VarNode("mat_k")->assert_is_op_input("matmul_v2", "Y")->AsInput();
    auto* mat_v =
        VarNode("mat_v")->assert_is_op_input("matmul_v2", "Y")->AsInput();

    auto* matmul_v2_0 = OpNode("matmul_v2_0", "matmul_v2")->AsIntermediate();
    auto* matmul_v2_0_out = VarNode("matmul_v2_0_out")
                                ->assert_is_op_output("matmul_v2", "Out")
                                ->AsIntermediate();
    PMNode* scale = nullptr;
    PMNode* scale_out = nullptr;
    if (with_inner_scale_) {
      scale =
          OpNode("scale", "scale")
              ->assert_op_attr_satisfied<float>(
                  "bias", [](float attr) { return (std::fabs(attr) < 1e-5); })
              ->AsIntermediate();
      scale_out = VarNode("scale_out")
                      ->assert_is_op_output("scale", "Out")
                      ->AsIntermediate();
    }

    PMNode* cast_1 = nullptr;
    PMNode* cast_1_out = nullptr;
    PMNode* cast_2 = nullptr;
    PMNode* cast_2_out = nullptr;
    if (with_cast_) {
      cast_1 = OpNode("cast_1", "cast")->AsIntermediate();
      cast_1_out = VarNode("cast_1_out")
                       ->assert_is_op_output("cast", "Out")
                       ->AsIntermediate();

      cast_2 = OpNode("cast_2", "cast")->AsIntermediate();
      cast_2_out = VarNode("cast_2_out")
                       ->assert_is_op_output("cast", "Out")
                       ->AsIntermediate();
    }

    auto* softmax = OpNode("softmax", "softmax")->AsIntermediate();
    auto* softmax_out = VarNode("softmax_out")
                            ->assert_is_op_output("softmax", "Out")
                            ->AsIntermediate();
    auto* matmul_v2_1 = OpNode("matmul_v2_1", "matmul_v2")->AsIntermediate();
    auto* matmul_v2_1_out = VarNode("matmul_v2_1_out")
                                ->assert_is_op_output("matmul_v2", "Out")
                                ->AsOutput();
    if (with_cast_) {
      if (with_inner_scale_) {
        *mat_q >> *matmul_v2_0 >> *matmul_v2_0_out >> *scale >> *scale_out >>
            *cast_1 >> *cast_1_out >> *softmax >> *softmax_out >> *cast_2 >>
            *cast_2_out >> *matmul_v2_1 >> *matmul_v2_1_out;
      } else {
        *mat_q >> *matmul_v2_0 >> *matmul_v2_0_out >> *cast_1 >> *cast_1_out >>
            *softmax >> *softmax_out >> *cast_2 >> *cast_2_out >>
            *matmul_v2_1 >> *matmul_v2_1_out;
      }
    } else {
      if (with_inner_scale_) {
        *mat_q >> *matmul_v2_0 >> *matmul_v2_0_out >> *scale >> *scale_out >>
            *softmax >> *softmax_out >> *matmul_v2_1 >> *matmul_v2_1_out;
      } else {
        *mat_q >> *matmul_v2_0 >> *matmul_v2_0_out >> *softmax >>
            *softmax_out >> *matmul_v2_1 >> *matmul_v2_1_out;
      }
    }
    *mat_k >> *matmul_v2_0;
    *mat_v >> *matmul_v2_1;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__matmul_scale_softmax_v1");
    op_desc.SetInput("mat_q", {matched.at("mat_q")->arg()->name});
    op_desc.SetInput("mat_k", {matched.at("mat_k")->arg()->name});
    op_desc.SetInput("mat_v", {matched.at("mat_v")->arg()->name});
    op_desc.SetOutput("Out", {matched.at("matmul_v2_1_out")->arg()->name});
    auto* scope = matched.at("softmax")->stmt()->op()->scope();

    auto mat_q_name = matched.at("mat_q")->arg()->name;
    auto mat_q_shape = scope->FindMutableTensor(mat_q_name)->dims();
    auto mat_k_name = matched.at("mat_k")->arg()->name;
    auto mat_k_shape = scope->FindMutableTensor(mat_k_name)->dims();
    auto mat_v_name = matched.at("mat_v")->arg()->name;
    auto mat_v_shape = scope->FindMutableTensor(mat_v_name)->dims();
    if (mat_q_shape.size() == 3) {
      CHECK_EQ(mat_q_shape.size(), 3) << mat_q_shape.size();
      CHECK_EQ(mat_k_shape.size(), 3) << mat_k_shape.size();
      CHECK_EQ(mat_v_shape.size(), 3) << mat_v_shape.size();
    } else {
      CHECK_EQ(mat_q_shape.size(), 4) << mat_q_shape.size();
      CHECK_EQ(mat_k_shape.size(), 4) << mat_k_shape.size();
      CHECK_EQ(mat_v_shape.size(), 4) << mat_v_shape.size();
    }

    if (with_inner_scale_) {
      auto* scale_instruct = matched.at("scale")->stmt();
      auto scale_op_desc = *scale_instruct->op_info();
      op_desc.SetAttr<float>("alpha", scale_op_desc.GetAttr<float>("scale"));
    } else {
      op_desc.SetAttr<float>("alpha", 1.0f);
    }

    auto* matmul_0_op_info = matched.at("matmul_v2_0")->stmt()->op_info();
    auto* matmul_1_op_info = matched.at("matmul_v2_1")->stmt()->op_info();
    std::vector<int> b_mm_trans_x_y;
    b_mm_trans_x_y.push_back(matmul_0_op_info->GetAttr<bool>("trans_x"));
    b_mm_trans_x_y.push_back(matmul_0_op_info->GetAttr<bool>("trans_y"));
    b_mm_trans_x_y.push_back(matmul_1_op_info->GetAttr<bool>("trans_x"));
    b_mm_trans_x_y.push_back(matmul_1_op_info->GetAttr<bool>("trans_y"));
    op_desc.SetAttr<std::vector<int>>("MatmulTransInfo", b_mm_trans_x_y);

    auto& valid_places =
        matched.at("matmul_v2_0")->stmt()->op()->valid_places();
    auto new_op = LiteOpRegistry::Global().Create(op_desc.Type());
    new_op->Attach(op_desc, scope);
    auto* new_op_node = graph->GraphCreateInstructNode(new_op, valid_places);

    DirectedLink(matched.at("mat_q"), new_op_node);
    DirectedLink(matched.at("mat_k"), new_op_node);
    DirectedLink(matched.at("mat_v"), new_op_node);
    DirectedLink(new_op_node, matched.at("matmul_v2_1_out"));
  }

 private:
  bool with_cast_;
  bool with_inner_scale_;
};
}  // namespace fusion

class XPUMatmulScaleSoftmaxV1FusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    if (GetBoolFromEnv("XPU_ENABLE_XTCL")) return;
    for (auto with_cast : {true, false}) {
      for (auto with_inner_scale : {true, false}) {
        fusion::MatmulScaleSoftmaxV1fuser fuser(with_cast, with_inner_scale);
        fuser(graph.get());
      }
    }
  }
};

/*
Fuse the following subgraph into __xpu__matmul_scale_softmax_v1 op for memory
reuse and optimization.

Original subgraph:

        Input1(Q)               Input2(K)
            |                      |
          Scale                  Scale
            \                      /
             \                    /
              \                  /
               \                /
                   matmul(_v2)
                       |
                      cast
                       |
                     softmax
                       |
                      cast          Input3(V)
                        \             /
                         \           /
                          matmul(_v2)
                              |
                            output

Ruse to:

        Input1(Q)    Input2(K)    Input3(V)
            \           |           /
          __xpu__matmul_scale_softmax_v1
                        |
                      output

*/

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__matmul_scale_softmax_v1_fuse_pass,
                  paddle::lite::mir::XPUMatmulScaleSoftmaxV1FusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__matmul_scale_softmax_v1");
