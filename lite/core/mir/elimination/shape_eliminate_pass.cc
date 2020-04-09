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

class Eliminator : public FuseBase {
 public:
  void BuildPattern() override {
    // the previous op's output need updat
    auto* pre_op = OpNode("preop")->assert_is_not_op_type("conditional_block");
    auto* input = VarNode("input")
                      ->assert_is_op_input("shape", "Input")
                      ->AsIntermediate();
    auto* shape_op =
        OpNode("shape", "shape")->assert_is_op("shape")->AsIntermediate();
    auto* out = VarNode("out")->assert_is_op_output("shape", "Out");
    LOG(INFO) << "shapeshape";

    *pre_op >> *input >> *shape_op >> *out;
  }

 private:
  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto& pre_op = matched.at("preop")->AsStmt();
    auto op_info = *pre_op.op_info();
    auto* shape_node = matched.at("shape");
    auto* scope = shape_node->stmt()->op()->scope();
    auto* in = matched.at("input");
    auto shape_in_tensor = scope->FindVar(in->arg()->name)->Get<lite::Tensor>();
    auto* out = matched.at("out");
    auto* shape_out_tensor =
        scope->FindVar(out->arg()->name)->GetMutable<lite::Tensor>();
    auto dim_data = shape_in_tensor.dims();
    std::vector<int64_t> shape_vec;
    shape_vec.push_back(static_cast<int64_t>(dim_data.size()));
    shape_out_tensor->Resize(shape_vec);
    auto* out_data = shape_out_tensor->mutable_data<int>();
    for (int i = 0; i < dim_data.size(); i++) {
      out_data[i] = dim_data[i];
    }
    op_info.UpdateAllOutputs(matched.at("input")->AsArg().name,
                             matched.at("out")->AsArg().name);
    pre_op.ResetOp(op_info, graph->valid_places());

    GraphSafeRemoveNodes(graph, {matched.at("shape")});

    IR_NODE_LINK_TO(matched.at("preop"), matched.at("out"));
  }
};

}  // namespace

class ShapeEliminatePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    Eliminator eliminator;
    eliminator(graph.get());
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(shape_eliminate_pass, paddle::lite::mir::ShapeEliminatePass)
    .BindTargets({TARGET(kAny)});
