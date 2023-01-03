// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/optimizer/mir/match_input_output_desc_pass.h"
#include <vector>
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher.h"

namespace paddle {
namespace lite {
namespace mir {

static Node* AddTranspose(Node* input_node,
                          const std::unique_ptr<SSAGraph>& graph,
                          Scope* scope,
                          const std::vector<lite_api::Place>& places) {
  // Create transpose out node
  auto input_name = input_node->arg()->name;
  auto output_name = input_name + "_transpose_out";
  auto output_node = graph->NewArgumentNode(output_name);
  output_node->arg()->type = input_node->arg()->type;
  // Create transpose out tensor
  scope->NewTensor(output_name);

  // Create transpose op_info
  cpp::OpDesc transpose_desc;
  std::string transpose_type("transpose");
  transpose_desc.SetType(transpose_type);
  transpose_desc.SetInput("X", {input_name});
  transpose_desc.SetOutput("Out", {output_name});
  transpose_desc.SetAttr("axis", std::vector<int>{0, 3, 1, 2});
  // Create transpose op
  auto transpose_op = LiteOpRegistry::Global().Create(transpose_type);
  transpose_op->Attach(transpose_desc, scope);
  // Create transpose node
  auto transpose_node = graph->GraphCreateInstructNode(transpose_op, places);

  // Reset output ops
  auto& output_op_nodes = input_node->outlinks;
  for (auto output_op_node = output_op_nodes.begin();
       output_op_node != output_op_nodes.end();
       output_op_node++) {
    OpInfo op_info = *((*output_op_node)->stmt()->op_info());
    op_info.UpdateAllInputs(input_name, output_name);
    (*output_op_node)->stmt()->ResetOp(op_info, places);
  }

  // Link node
  input_node->outlinks.clear();
  IR_NODE_LINK_TO(input_node, transpose_node);
  IR_NODE_LINK_TO(transpose_node, output_node);
  for (auto output_op_node = output_op_nodes.begin();
       output_op_node != output_op_nodes.end();
       output_op_node++) {
    auto& in_nodes = (*output_op_node)->inlinks;
    for (auto in_node = in_nodes.begin(); in_node != in_nodes.end();
         in_node++) {
      if ((*in_node) == input_node) {
        in_nodes.erase(in_node);
        break;
      }
    }
    IR_NODE_LINK_TO(output_node, (*output_op_node));
  }

  return output_node;
}

static Node* AddTranspose(Node* input_node,
                          const std::unique_ptr<SSAGraph>& graph,
                          Scope* scope,
                          const std::vector<lite_api::Place>& places,
                          float scale,
                          int zero_point) {
  return nullptr;
}

void MatchInputOutputDescPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  LOG(INFO) << "--- MatchInputOutputDescPass, 0";
  Scope* scope = nullptr;
  std::vector<lite_api::Place> places;
  for (auto& node : graph->nodes()) {
    if (node.IsStmt()) {
      auto op = node.stmt()->op();
      scope = op->scope();
      places = op->valid_places();
      break;
    }
  }
  CHECK(scope);

  std::map<std::string, lite_api::InputDesc> input_descs =
      Context<TargetType::kHost>::InputDesc(scope);
  std::map<std::string, lite_api::OutputDesc> output_descs =
      Context<TargetType::kHost>::OutputDesc(scope);

  for (auto iter : input_descs) {
    auto input_name = iter.first;
    auto input_desc = iter.second;
    auto input_node = graph->RetrieveArgument(input_name);
    CHECK(input_node) << "Not find node(" << input_name
                      << "), please check your input_desc.";
    auto input_arg = input_node->arg();

    // Update precision
    auto input_type = input_arg->type;
    auto precision = input_desc.precision;
    if (precision == PRECISION(kUInt8)) {
      precision = PRECISION(kInt8);
    } else if (precision == PRECISION(kUnk)) {
      precision = input_type->precision;
    }
    input_arg->type = Type::GetTensorTy(
        input_type->target(), precision, input_type->layout());

    // Update scale/zero_pint
    // if (precision == PRECISION(kUInt8)) {
    //   auto var = scope->Var("EXTRA_PROPERTY");
    //   auto data = var->GetMutable<std::string>();
    //   *data = string_format(
    //       "QUALCOMM_QNN_INPUT_QUANT_PARAMS=0,%f,%d;QUALCOMM_QNN_SKIP_"
    //       "SYMM2ASYMM=1",
    //       input_desc.quant_scale,
    //       input_desc.quant_zero_point);
    // }

    // // Transpose: nchw -> nhwc
    // auto layout=input_desc.layout;
    // CHECK(layout==DATALAYOUT(kNCHW) or layout==DATALAYOUT(kNHWC))<<"NNAdapter
    // only support input layout is nchw or nhwc, but received is
    // "<<DataLayoutToStr(layout);
    // Node* transpose_out_node=input_node;
    // if (layout == DATALAYOUT(kNHWC)) {
    //   transpose_out_node=AddTranspose(input_node, graph, scope, places);
    // }

    // // Resize
    // auto resize_type=input_desc.resize_type;
    // Node* resize_out_node=transpose_out_node;
    // if(!resize_type.empty()){
    //   CHECK(resize_type=="linear" or resize_type=="nearest")<<"NNAdapter only
    //   support input resize_type is linear or nearest, but received is
    //   "<<resize_type;
    //   int target_h=input_desc.resize_target_height;
    //   int target_w=input_desc.resize_target_width;
    //   CHECK_GT(target_h, 0);
    //   CHECK_GT(target_w, 0);
    //   resize_out_node=AddResize(input_node, graph, scope, places,
    //   resize_type);
    // }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(match_input_output_desc_pass,
                  paddle::lite::mir::MatchInputOutputDescPass)
    .BindTargets({TARGET(kNNAdapter)});
