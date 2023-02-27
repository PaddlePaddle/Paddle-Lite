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

class XPUEncoderAdaptiveSeqlenFuser : public FuseBase {
 public:
  explicit XPUEncoderAdaptiveSeqlenFuser(
      const std::string& matmul_type = "matmul", bool with_bias = false)
      : matmul_type_(matmul_type), with_bias_(with_bias) {}

  void BuildPattern() override {
    auto* mask = VarNode("mask")
                     ->assert_is_op_input(matmul_type_, "X")
                     ->assert_is_op_input(matmul_type_, "Y")
                     ->AsInput();
    auto* matmul = OpNode("matmul", matmul_type_)->AsIntermediate();
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
    auto* stack_out =
        VarNode("stack_out")->assert_is_op_output("stack", "Y")->AsOutput();
    *mask >> *matmul >> *matmul_out >> *scale >> *scale_out >> *stack >>
        *stack_out;
    PMNode* ewadd_in = nullptr;
    PMNode* ewadd = nullptr;
    PMNode* ewadd_out = nullptr;
    if (with_bias_) {
      stack_out->assert_is_op_input("elementwise_add", "X")->AsIntermediate();
      ewadd_in = VarNode("ewadd_in")
                     ->assert_is_op_input("elementwise_add", "Y")
                     ->AsInput();
      ewadd = OpNode("ewadd", "elementwise_add")->AsIntermediate();
      ewadd_out = VarNode("ewadd_out")
                      ->assert_is_op_output("elementwise_add", "Out")
                      ->assert_is_op_input("__xpu__encoder", "Mask")
                      ->AsOutput();
      *stack_out >> *ewadd >> *ewadd_out;
      *ewadd_in >> *ewadd;
    } else {
      stack_out->assert_is_op_input("__xpu__encoder", "Mask");
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    VLOG(3) << "______xpu_encoder_adaptive_seqlen______";
    auto* scope = matched.at("matmul")->stmt()->op()->scope();
    auto valid_places = matched.at("matmul")->stmt()->op()->valid_places();
    auto* mask_out_node =
        (with_bias_) ? matched.at("ewadd_out") : matched.at("stack_out");
    std::string mask_name = matched.at("mask")->arg()->name;

    // arg seq_lod (reuse mask_out_node)
    auto* seq_lod_node = mask_out_node;
    seq_lod_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kNCHW));

    // add new arg seq_len
    std::string seq_len_name = mask_name + "_seq_len";
    auto* seq_len_node = graph->NewArgumentNode(seq_len_name);
    seq_len_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kInt64), DATALAYOUT(kNCHW));
    scope->NewTensor(seq_len_name);

    // add new arg pad_seq_len
    std::string pad_seq_len_name = mask_name + "_pad_seq_len";
    auto* pad_seq_len_node = graph->NewArgumentNode(pad_seq_len_name);
    pad_seq_len_node->arg()->type = LiteType::GetTensorTy(
        TARGET(kHost), PRECISION(kInt32), DATALAYOUT(kNCHW));
    scope->NewTensor(pad_seq_len_name);

    // set seq_lod_op
    cpp::OpDesc seq_lod_op_desc;
    seq_lod_op_desc.mutable_inputs()->clear();
    seq_lod_op_desc.mutable_outputs()->clear();
    seq_lod_op_desc.SetType("calculate_lod");
    seq_lod_op_desc.SetInput("Mask", {mask_name});
    seq_lod_op_desc.SetOutput("SeqLod", {seq_lod_node->arg()->name});
    seq_lod_op_desc.SetOutput("SeqLen", {seq_len_name});
    seq_lod_op_desc.SetOutput("PadSeqLen", {pad_seq_len_name});
    auto seq_lod_op = LiteOpRegistry::Global().Create("calculate_lod");
    seq_lod_op->Attach(seq_lod_op_desc, scope);
    seq_lod_op->SetValidPlaces(valid_places);
    auto* seq_lod_op_node =
        graph->GraphCreateInstructNode(seq_lod_op, valid_places);
    DirectedLink(matched.at("mask"), seq_lod_op_node);
    DirectedLink(seq_lod_op_node, pad_seq_len_node);
    DirectedLink(seq_lod_op_node, seq_lod_node);
    DirectedLink(seq_lod_op_node, seq_len_node);
    for (auto* encoder_node :
         mask_out_node->outlinks) {  // TODO(TingShen): check is encoder
      auto* encoder_instruct = encoder_node->stmt();
      auto* encoder_op_desc = encoder_instruct->mutable_op_info();
      auto encoder_op = encoder_instruct->op();
      Node* encoder_input_node;
      for (auto* encoder_in_node : encoder_node->inlinks) {
        if (encoder_in_node->arg()->name ==
            encoder_op_desc->Input("Input")[0]) {
          encoder_input_node = encoder_in_node;
        }
      }
      std::string encoder_input_name = encoder_input_node->arg()->name;
      // add new arg seq_out
      std::string seq_out_name = encoder_input_name + "_squeezed";
      auto* seq_out_node = graph->NewArgumentNode(seq_out_name);
      seq_out_node->arg()->type = LiteType::GetTensorTy(
          TARGET(kXPU), PRECISION(kFloat), DATALAYOUT(kNCHW));
      scope->NewTensor(seq_out_name);

      // add sequence_unpad op
      cpp::OpDesc seq_unpad_op_desc;
      seq_unpad_op_desc.mutable_inputs()->clear();
      seq_unpad_op_desc.mutable_outputs()->clear();
      seq_unpad_op_desc.SetType("sequence_unpad");
      seq_unpad_op_desc.SetInput("X", {encoder_input_name});
      seq_unpad_op_desc.SetInput("Length", {seq_len_name});
      seq_unpad_op_desc.SetOutput("Out", {seq_out_name});
      auto seq_unpad_op = LiteOpRegistry::Global().Create("sequence_unpad");
      seq_unpad_op->Attach(seq_unpad_op_desc, scope);
      seq_unpad_op->SetValidPlaces(valid_places);
      auto* seq_unpad_op_node =
          graph->GraphCreateInstructNode(seq_unpad_op, valid_places);

      // set encoder op
      encoder_op_desc->SetInput("SeqLod", {seq_lod_node->arg()->name});
      encoder_op_desc->SetInput("PadSeqLen", {pad_seq_len_name});
      encoder_op_desc->SetInput("Input", {seq_out_name});
      encoder_op_desc->SetAttr<bool>("do_padding", true);
      encoder_op_desc->SetAttr<bool>("adaptive_seqlen", true);
      if (with_bias_) {
        encoder_op_desc->SetInput("Mask",
                                  {matched.at("ewadd_in")->arg()->name});
      } else {
        encoder_op_desc->SetInput("Mask", {});
      }

      DirectedLink(encoder_input_node, seq_unpad_op_node);
      DirectedLink(seq_unpad_op_node, seq_out_node);
      DirectedLink(pad_seq_len_node, encoder_node);
      // DirectedLink(seq_lod_node, encoder_node);
      DirectedLink(seq_out_node, encoder_node);

      RemoveDirectedLink(encoder_input_node, encoder_node);
    }
  }

 private:
  std::string matmul_type_;
  bool with_bias_;
};

}  // namespace fusion

class XPUEncoderAdaptiveSeqlenFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    bool adaptive_seqlen = false;
#ifdef LITE_WITH_XPU
    //  To suppress linkage error, we use #ifdef here.
    adaptive_seqlen =
        lite::TargetWrapperXPU::xpu_runtime_ptr->multi_encoder_adaptive_seqlen;
    VLOG(3) << "adaptive_seqlen: " << adaptive_seqlen;
#endif
    if (adaptive_seqlen == false) return;

    std::vector<std::string> matmul_types{"matmul", "matmul_v2"};
    std::vector<bool> with_biass{false};
    for (auto& matmul_type : matmul_types) {
      for (auto with_bias : with_biass) {
        fusion::XPUEncoderAdaptiveSeqlenFuser fuser(matmul_type, with_bias);
        fuser(graph.get());
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__encoder_adaptive_seqlen_fuse_pass,
                  paddle::lite::mir::XPUEncoderAdaptiveSeqlenFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("sequence_unpad")
    .BindKernel("calculate_lod");
