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

/* support adaptive seq len for mrc          */
/*                in_Input      in_Mask      */
/*                    |             |        */
/*                    |             |        */
/*                    |           matmul     */
/*                    |             |        */
/*                    |             |        */
/*                    |           scale      */
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
/*                    |        |             */
/*                    | xpu_adaptive_mask    */
/*                    |        |     |       */
/*          sequence_unpad<--Lenght  |       */
/*                    |              |       */
/*                    |            PadSeqLen */
/*                    |            SeqLod    */
/*                    |            /         */
/*                    |          /           */
/*                    |        /             */
/*                xpu_encoder                */
/*                    |                      */
/*                    |                      */
/*                out_Output                 */
/*-------------------------------------------*/

class XPUMultiEncoderAdaptiveSeqlenV3Fuser : public FuseBase {
 public:
  explicit XPUMultiEncoderAdaptiveSeqlenV3Fuser(
      const std::string& matmul_type = "matmul")
      : matmul_type_(matmul_type) {}

  void BuildPattern() override {
    auto* mask = VarNode("mask")
                     ->assert_is_op_input(matmul_type_, "X")
                     ->assert_is_op_input(matmul_type_, "Y");
    auto* matmul = OpNode(matmul_type_, matmul_type_)->AsIntermediate();
    auto* matmul_out = VarNode("matmul_out")
                           ->assert_is_op_input("scale", "X")
                           ->assert_is_op_output(matmul_type_, "Out")
                           ->AsIntermediate();
    auto* scale =
        OpNode("scale", "scale")
            ->assert_op_attr<bool>("bias_after_scale", false)
            ->assert_op_attr_satisfied<float>(
                "bias",
                [](float attr) { return (std::fabs(attr + 1.0) < 1e-5); })
            ->assert_op_attr_satisfied<float>(
                "scale",
                [](float attr) { return (std::fabs(attr - 10000.0) < 1e-5); })
            ->AsIntermediate();
    auto* scale_out = VarNode("scale_out")
                          ->assert_is_op_input("stack", "X")
                          ->assert_is_op_output("scale", "Out")
                          ->AsIntermediate();
    auto* stack = OpNode("stack", "stack")->AsIntermediate();
    auto* stack_out = VarNode("stack_out")
                          ->assert_is_op_input("__xpu__multi_encoder", "Mask")
                          ->assert_is_op_output("stack", "Y")
                          ->AsIntermediate();
    auto* encoder_input =
        VarNode("encoder_input")
            ->assert_is_op_input("__xpu__multi_encoder", "Input");
    auto* xpu_encoder = OpNode("xpu_encoder", "__xpu__multi_encoder")
                            ->assert_op_attr<bool>("adaptive_seqlen", true);

    *mask >> *matmul >> *matmul_out >> *scale >> *scale_out >> *stack >>
        *stack_out >> *xpu_encoder;
    *encoder_input >> *xpu_encoder;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    auto* encoder_instruct = matched.at("xpu_encoder")->stmt();
    auto encoder_op_desc = encoder_instruct->mutable_op_info();
    auto encoder_op = encoder_instruct->op();
    auto* scope = encoder_op->scope();

    // add new arg seq_lod
    std::string stack_out_name = matched.at("stack_out")->arg()->name;
    std::string xpu_mask_adaptive_seq_lod_name = stack_out_name + "_seq_lod";
    auto* xpu_mask_adaptive_seq_lod_node =
        graph->NewArgumentNode(xpu_mask_adaptive_seq_lod_name);
    xpu_mask_adaptive_seq_lod_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kNCHW));
    scope->NewTensor(xpu_mask_adaptive_seq_lod_name);
    // add new arg pad_seq_len, store max padded length
    std::string xpu_mask_adaptive_pad_seq_len_name =
        stack_out_name + "_pad_seq_len";
    auto* xpu_mask_adaptive_pad_seq_len_node =
        graph->NewArgumentNode(xpu_mask_adaptive_pad_seq_len_name);
    xpu_mask_adaptive_pad_seq_len_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kNCHW));
    scope->NewTensor(xpu_mask_adaptive_pad_seq_len_name);
    // add new arg length, for sequence_unpad, store length in batch
    std::string xpu_mask_adaptive_seq_len_name = stack_out_name + "_seq_length";
    auto* xpu_mask_adaptive_seq_len_node =
        graph->NewArgumentNode(xpu_mask_adaptive_seq_len_name);
    xpu_mask_adaptive_seq_len_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kInt64), DATALAYOUT(kNCHW));
    scope->NewTensor(xpu_mask_adaptive_seq_len_name);

    // add new packed input of encoder
    std::string orig_encoder_input_name =
        matched.at("encoder_input")->arg()->name;
    std::string packed_encoder_input_name =
        orig_encoder_input_name + "_vsl_packed";
    auto* packed_encoder_input_node =
        graph->NewArgumentNode(packed_encoder_input_name);
    packed_encoder_input_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
    scope->NewTensor(packed_encoder_input_name);

    // create xpu_mask_adaptive op to set lod
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__mask_adaptive");
    op_desc.SetInput("Mask", {matched.at("mask")->arg()->name});
    op_desc.SetOutput(
        "Length",
        {xpu_mask_adaptive_seq_len_name});  // length for sequence_unpad op
    op_desc.SetOutput("SeqLod",
                      {xpu_mask_adaptive_seq_lod_name});  // lod for encoder op
    op_desc.SetOutput("PadSeqLen", {xpu_mask_adaptive_pad_seq_len_name});
    auto xpu_mask_adaptive_op =
        LiteOpRegistry::Global().Create("__xpu__mask_adaptive");
    auto& valid_places = encoder_op->valid_places();
    xpu_mask_adaptive_op->Attach(op_desc, scope);
    auto* xpu_mask_adaptive_node =
        graph->GraphCreateInstructNode(xpu_mask_adaptive_op, valid_places);

    // create sequence_unpad to pack the encoder input
    cpp::OpDesc sequence_unpad_op_desc;
    sequence_unpad_op_desc.SetType("sequence_unpad");
    sequence_unpad_op_desc.SetInput("X",
                                    {matched.at("encoder_input")->arg()->name});
    sequence_unpad_op_desc.SetInput("Length", {xpu_mask_adaptive_seq_len_name});
    sequence_unpad_op_desc.SetOutput("Out", {packed_encoder_input_name});
    auto sequence_unpad_op = LiteOpRegistry::Global().Create("sequence_unpad");
    sequence_unpad_op->Attach(sequence_unpad_op_desc, scope);
    auto* sequence_unpad_node =
        graph->GraphCreateInstructNode(sequence_unpad_op, valid_places);

    encoder_op_desc->SetInput("Input", {packed_encoder_input_name});
    encoder_op_desc->SetInput("SeqLod", {xpu_mask_adaptive_seq_lod_name});
    encoder_op_desc->SetInput("PadSeqLen",
                              {xpu_mask_adaptive_pad_seq_len_name});
    auto updated_encoder_op_desc = *encoder_instruct->mutable_op_info();
    encoder_instruct->ResetOp(updated_encoder_op_desc, valid_places);

    RemoveDirectedLink(matched.at("encoder_input"), matched.at("xpu_encoder"));
    DirectedLink(matched.at("mask"), xpu_mask_adaptive_node);
    DirectedLink(xpu_mask_adaptive_node, xpu_mask_adaptive_seq_lod_node);
    DirectedLink(xpu_mask_adaptive_node, xpu_mask_adaptive_pad_seq_len_node);
    DirectedLink(xpu_mask_adaptive_node, xpu_mask_adaptive_seq_len_node);
    DirectedLink(xpu_mask_adaptive_seq_lod_node, matched.at("xpu_encoder"));
    DirectedLink(xpu_mask_adaptive_pad_seq_len_node, matched.at("xpu_encoder"));
    DirectedLink(xpu_mask_adaptive_seq_len_node, sequence_unpad_node);
    DirectedLink(matched.at("encoder_input"), sequence_unpad_node);
    DirectedLink(sequence_unpad_node, packed_encoder_input_node);
    DirectedLink(packed_encoder_input_node, matched.at("xpu_encoder"));
  }

 private:
  std::string matmul_type_;
};

}  // namespace fusion

class XPUMultiEncoderAdaptiveSeqlenV3FusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    std::vector<std::string> matmul_types{"matmul", "matmul_v2"};
    for (auto& matmul_type : matmul_types) {
      fusion::XPUMultiEncoderAdaptiveSeqlenV3Fuser fuser(matmul_type);
      fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__multi_encoder_adaptive_seqlen_v3_fuse_pass,
                  paddle::lite::mir::XPUMultiEncoderAdaptiveSeqlenV3FusePass)
    .BindTargets({TARGET(kXPU)});
