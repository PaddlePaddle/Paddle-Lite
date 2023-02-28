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

/* support adaptive seq len for bert/ernie   */
/*                in_Input      in_Mask      */
/*                    |             |        */
/*                    |             |        */
/*                xpu_embedding   matmul     */
/*                    |             |        */
/*                    |             |        */
/*                 layer_norm     scale      */
/*                    |           /          */
/*                    |        stack         */
/*                    |         |            */
/*                    |        /             */
/*                    |      /               */
/*                xpu_encoder                */
/*                    |                      */
/*                    |                      */
/*                out_Output                 */
/*-------------------------------------------*/
/* After the pass apply:                     */
/*                in_Input  in_Mask          */
/*                    |        |             */
/*                    |       /              */
/*                xpu_embedding              */
/*                    |       \              */
/*                    |     SeqLod           */
/*                    |        |             */
/*                 layer_norm  |             */
/*                    |        |             */
/*                    |       /              */
/*                xpu_encoder                */
/*                    |                      */
/*                    |                      */
/*                out_Output                 */
/*-------------------------------------------*/

class XPUMultiEncoderAdaptiveSeqlenFuser : public FuseBase {
 public:
  explicit XPUMultiEncoderAdaptiveSeqlenFuser(
      const std::string& matmul_type = "matmul", bool pre_ln = false)
      : matmul_type_(matmul_type), pre_ln_(pre_ln) {}

  void BuildPattern() override {
    auto* mask = VarNode("mask")
                     ->assert_is_op_input(matmul_type_, "X")
                     ->assert_is_op_input(matmul_type_, "Y");
    auto* matmul = OpNode(matmul_type_, matmul_type_)->AsIntermediate();
    auto* matmul_out = VarNode("matmul_out")
                           ->assert_is_op_input("scale", "X")
                           ->assert_is_op_output(matmul_type_, "Out")
                           ->AsIntermediate();
    auto* scale = OpNode("scale", "scale")->AsIntermediate();
    auto* scale_out = VarNode("scale_out")
                          ->assert_is_op_input("stack", "X")
                          ->assert_is_op_output("scale", "Out")
                          ->AsIntermediate();
    auto* stack = OpNode("stack", "stack")->AsIntermediate();
    auto* stack_out = VarNode("stack_out")
                          ->assert_is_op_input("__xpu__multi_encoder", "Mask")
                          ->assert_is_op_output("stack", "Y")
                          ->AsIntermediate();
    auto* xpu_embedding =
        OpNode("xpu_embedding", "__xpu__embedding_with_eltwise_add");

    PMNode* embedding_out = nullptr;
    PMNode* layer_norm = nullptr;
    PMNode* layer_norm_out = nullptr;

    if (pre_ln_) {
      embedding_out = VarNode("embedding_out")
                          ->assert_is_op_output(
                              "__xpu__embedding_with_eltwise_add", "Output")
                          ->assert_is_op_input("__xpu__multi_encoder", "Input");
    } else {
      embedding_out = VarNode("embedding_out")
                          ->assert_is_op_output(
                              "__xpu__embedding_with_eltwise_add", "Output")
                          ->assert_is_op_input("layer_norm", "X");
      layer_norm = OpNode("layer_norm", "layer_norm");
      layer_norm_out =
          VarNode("layer_norm_out")
              ->assert_is_op_output("layer_norm", "Y")
              ->assert_is_op_input("__xpu__multi_encoder", "Input");
    }
    auto* xpu_encoder = OpNode("xpu_encoder", "__xpu__multi_encoder")
                            ->assert_op_attr<bool>("adaptive_seqlen", true);
    if (pre_ln_) {
      xpu_encoder->assert_op_attr<bool>("norm_before", true);
      *xpu_embedding >> *embedding_out >> *xpu_encoder;
    } else {
      *xpu_embedding >> *embedding_out >> *layer_norm >> *layer_norm_out >>
          *xpu_encoder;
    }

    *mask >> *matmul >> *matmul_out >> *scale >> *scale_out >> *stack >>
        *stack_out >> *xpu_encoder;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto* embedding_instruct = matched.at("xpu_embedding")->stmt();
    auto embedding_op_desc = *embedding_instruct->mutable_op_info();
    auto embedding_op = embedding_instruct->op();
    auto* scope = embedding_op->scope();
    auto* encoder_instruct = matched.at("xpu_encoder")->stmt();
    auto encoder_op_desc = *encoder_instruct->mutable_op_info();
    auto encoder_op = encoder_instruct->op();

    // add new arg seq_lod
    std::string embedding_out_name = matched.at("embedding_out")->arg()->name;
    std::string embedding_seq_lod_name = embedding_out_name + "_seq_lod";
    auto* embedding_seq_lod_node =
        graph->NewArgumentNode(embedding_seq_lod_name);
    embedding_seq_lod_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kNCHW));
    scope->NewTensor(embedding_seq_lod_name);
    // add new arg pad_seq_len
    std::string embedding_pad_seq_len_name =
        embedding_out_name + "_pad_seq_len";
    auto* embedding_pad_seq_len_node =
        graph->NewArgumentNode(embedding_pad_seq_len_name);
    embedding_pad_seq_len_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kNCHW));
    scope->NewTensor(embedding_pad_seq_len_name);

    embedding_op_desc.SetOutput("SeqLod", {embedding_seq_lod_name});
    embedding_op_desc.SetOutput("PadSeqLen", {embedding_pad_seq_len_name});
    encoder_op_desc.SetInput("SeqLod", {embedding_seq_lod_name});
    encoder_op_desc.SetInput("PadSeqLen", {embedding_pad_seq_len_name});
    embedding_op_desc.SetInput("Mask", {matched.at("mask")->arg()->name});

    embedding_instruct->ResetOp(embedding_op_desc,
                                embedding_op->valid_places());
    encoder_instruct->ResetOp(encoder_op_desc, encoder_op->valid_places());
    DirectedLink(matched.at("xpu_embedding"), embedding_seq_lod_node);
    DirectedLink(matched.at("xpu_embedding"), embedding_pad_seq_len_node);
    DirectedLink(matched.at("mask"), matched.at("xpu_embedding"));
    DirectedLink(embedding_seq_lod_node, matched.at("xpu_encoder"));
    DirectedLink(embedding_pad_seq_len_node, matched.at("xpu_encoder"));
  }

 private:
  std::string matmul_type_;
  bool pre_ln_;
};

}  // namespace fusion

class XPUMultiEncoderAdaptiveSeqlenFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    std::vector<std::string> matmul_types{"matmul", "matmul_v2"};
    std::vector<bool> pre_lns{true, false};
    for (auto& matmul_type : matmul_types) {
      for (auto pre_ln : pre_lns) {
        fusion::XPUMultiEncoderAdaptiveSeqlenFuser fuser(matmul_type, pre_ln);
        fuser(graph.get());
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__multi_encoder_adaptive_seqlen_fuse_pass,
                  paddle::lite::mir::XPUMultiEncoderAdaptiveSeqlenFusePass)
    .BindTargets({TARGET(kXPU)});
