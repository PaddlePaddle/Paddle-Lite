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

#include <memory>
#include <string>
#include "lite/backends/xpu/math.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

/*
*  Match this graph (q.dims = k.dims = v.dims)
    q
*    \
*     \
*    transpose2    k
*          \      /
*           \    /
*           matmul
*             |
*             |
*           scale
*             |
*             |
*          softmax     v
*             \       /
*              \     /
*              matmul
*                |
*                |
*            transpose2
*                |
*                |
*               qkv
*/

class XpuQkvFuser : public FuseBase {
public:
    void BuildPattern() override {
        auto* tx = VarNode("t")->assert_is_op_input("transpose2", "X")->AsInput();
        auto* transpose = OpNode("transpose", "transpose2");
        auto* t_output = VarNode("t_output")->assert_is_op_output("transpose2", "XShape");

        auto* mx1 = VarNode("m1x")->assert_is_op_input("matmul_v2", "X")->AsInput();
        auto* my1 = VarNode("m1y")->assert_is_op_input("matmul_v2", "Y"); 
        auto* mul1 = OpNode("mul1", "matmul_v2");

        auto* sx = VarNode("s")->assert_is_op_input("scale", "X"); 
        auto* scale = OpNode("scale", "scale");

        auto* smx = VarNode("sm")->assert_is_op_input("softmax", "X"); 
        auto* softmax = OpNode("softmax", "softmax");

        auto* mx2 = VarNode("m2x")->assert_is_op_input("matmul_v2", "X"); 
        auto* my2 = VarNode("m2y")->assert_is_op_input("matmul_v2", "Y")->AsInput(); 
        auto* mul2 = OpNode("mul2", "matmul_v2");

        auto* output = VarNode("output")->assert_is_op_output("matmul_v2", "Out")->AsOutput();

        *tx >> *transpose >> *my1;
        *transpose >> *t_output;
        *mx1 >> *mul1;
        *my1 >> *mul1;
        *mul1 >> *sx >> *scale;
        *scale >> *smx >> *softmax;
        *softmax >> *mx2;
        *mx2 >> *mul2;
        *my2 >> *mul2;
        *mul2 >> *output;

        transpose->AsIntermediate();
        t_output->AsIntermediate();
        my1->AsIntermediate();
        mul1->AsIntermediate();
        sx->AsIntermediate();
        scale->AsIntermediate();
        smx->AsIntermediate();
        softmax->AsIntermediate();
        mx2->AsIntermediate();
        mul2->AsIntermediate();
    }

    void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
        cpp::OpDesc op_desc;
        op_desc.SetType("__xpu__qkv_attention");

        // set input
        auto input_name_q = matched.at("m1x")->arg()->name;
        op_desc.SetInput("input_q", {input_name_q});
        auto input_name_k = matched.at("t")->arg()->name;
        op_desc.SetInput("input_k", {input_name_k});
        auto input_name_v = matched.at("m2y")->arg()->name;
        op_desc.SetInput("input_v", {input_name_v});

        // set output
        op_desc.SetOutput("output", {matched.at("output")->arg()->name});
        
        // set params
        auto* scale_instruct = matched.at("scale")->stmt();
        auto scale_op_desc = *scale_instruct->op_info();
        op_desc.SetAttr<float>("scale_scale", scale_op_desc.GetAttr<float>("scale"));
        op_desc.SetAttr<float>("scale_bias", scale_op_desc.GetAttr<float>("bias"));

        auto qkv_attention_op =
            LiteOpRegistry::Global().Create("__xpu__qkv_attention");
        auto mul1 = matched.at("mul1")->stmt()->op();
        auto& valid_places = mul1->valid_places();
        auto* scope = mul1->scope();
        qkv_attention_op->Attach(op_desc, scope);
        auto* new_op_node =
            graph->GraphCreateInstructNode(qkv_attention_op, valid_places);

        DirectedLink(matched.at("m1x"), new_op_node);
        DirectedLink(matched.at("t"), new_op_node);
        DirectedLink(matched.at("m2y"), new_op_node);
        //DirectedLink(matched.at("m2y"), new_op_node);
        DirectedLink(new_op_node, matched.at("output"));
    }
};
}  // namespace fusion

class XpuQkvFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
      fusion::XpuQkvFuser qkv_fuse;
      qkv_fuse(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__qkv_fuse_pass,
                  paddle::lite::mir::XpuQkvFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__qkv_attention");