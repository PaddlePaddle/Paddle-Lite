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

#include "lite/core/mir/pass.h"
#include "lite/core/mir/pass_registry.h"
#include "lite/core/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {

namespace {

class CastEliminator : public FuseBase {
 public:
  explicit CastEliminator(const int dtype) : dtype_(dtype) {}
  void BuildPattern() override {
    // the previous op's output need updat
    auto* pre_op = OpNode("preop")
                       ->assert_is_not_op_type("conditional_block")
                       ->assert_is_not_op_type("cast");

    auto* x = VarNode("x")->assert_is_op_input("cast", "X");
    auto* cast_op = OpNode("cast", "cast")
                        ->assert_op_attr<int>("in_dtype", dtype_)
                        ->assert_op_attr<int>("out_dtype", dtype_);
    auto* out = VarNode("out")->assert_is_op_output("cast", "Out");
    *pre_op >> *x >> *cast_op >> *out;
    // The pre_op will be eliminated, and a new output-updated op will insert.
  }

 private:
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto& pre_op = matched.at("preop")->AsStmt();
    auto op_info = *pre_op.op_info();

    op_info.UpdateAllOutputs(matched.at("x")->AsArg().name,
                             matched.at("out")->AsArg().name);
    pre_op.ResetOp(op_info, graph->valid_places());
    auto& cast_op = matched.at("cast")->AsStmt();
    auto cast_op_desc = *cast_op.op_info();
    auto in_dtype = cast_op_desc.GetAttr<int>("in_dtype");
    auto out_dtype = cast_op_desc.GetAttr<int>("out_dtype");
    // ====================== DEBUG INFO =========================
    VLOG(6) << "in_dtype : " << in_dtype;
    VLOG(6) << "out_dtype : " << out_dtype;
    // ====================== DEBUG END  =========================
    GraphSafeRemoveNodes(graph, {matched.at("cast")});

    IR_NODE_LINK_TO(matched.at("preop"), matched.at("out"));
  }
  int dtype_ = -1;
};

}  // namespace

class IdentityCastEliminatePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    // const int BOOL = 0;
    // const int INT16 = 1;
    const int INT32 = 2;
    // const int INT64 = 3;
    // const int FP16 = 4;
    // const int FP32 = 5;
    // const int FP64 = 6;
    // const int UINT8 = 20;
    // const int INT8 = 21;
    CastEliminator eliminator_int32(INT32);
    eliminator_int32(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(identity_cast_eliminate_pass,
                  paddle::lite::mir::IdentityCastEliminatePass)
    .BindTargets({TARGET(kMLU)});
