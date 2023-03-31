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

class XPUEmbeddingUnpadFuser : public FuseBase {
 public:
  explicit XPUEmbeddingUnpadFuser(bool with_ln = true) : with_ln_(with_ln) {}
  void BuildPattern() override {
    auto* embedding = OpNode("embedding", "__xpu__embedding_with_eltwise_add");
    auto* embedding_out =
        VarNode("embedding_out")
            ->assert_is_op_output("__xpu__embedding_with_eltwise_add",
                                  "Output");
    auto* seq_unpad = OpNode("seq_unpad", "sequence_unpad")->AsIntermediate();
    auto* seq_unpad_out = VarNode("seq_unpad_out")
                              ->assert_is_op_input("__xpu__encoder", "Input")
                              ->assert_is_op_output("sequence_unpad", "Out")
                              ->AsIntermediate();
    auto* xpu_encoder = OpNode("xpu_encoder", "__xpu__encoder")
                            ->assert_op_attr<bool>("adaptive_seqlen", true);
    auto* seq_lod = VarNode("seq_lod")
                        ->assert_is_op_input("__xpu__encoder", "SeqLod")
                        ->assert_is_op_output("calculate_lod", "SeqLod");
    auto* pad_seq_len = VarNode("pad_seq_len")
                            ->assert_is_op_input("__xpu__encoder", "PadSeqLen")
                            ->assert_is_op_output("calculate_lod", "PadSeqLen");
    *seq_lod >> *xpu_encoder;
    *pad_seq_len >> *xpu_encoder;
    if (with_ln_) {
      embedding_out->assert_is_op_input("layer_norm", "X");
      auto* layer_norm = OpNode("layer_norm", "layer_norm");
      auto* layer_norm_out = VarNode("layer_norm_out")
                                 ->assert_is_op_output("layer_norm", "Y")
                                 ->assert_is_op_input("sequence_unpad", "X");
      *embedding >> *embedding_out >> *layer_norm >> *layer_norm_out >>
          *seq_unpad >> *seq_unpad_out >> *xpu_encoder;
    } else {
      embedding_out->assert_is_op_input("sequence_unpad", "X");
      *embedding >> *embedding_out >> *seq_unpad >> *seq_unpad_out >>
          *xpu_encoder;
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    VLOG(3) << "______xpu_embedding_unpad_fuse_pass______";
    cpp::OpDesc embedding_op_desc = *matched.at("embedding")->stmt()->op_info();
    cpp::OpDesc encoder_op_desc = *matched.at("xpu_encoder")->stmt()->op_info();
    auto valid_places = matched.at("embedding")->stmt()->op()->valid_places();
    auto embedding_instruct = matched.at("embedding")->stmt();
    embedding_op_desc.SetInput("SeqLod", {matched.at("seq_lod")->arg()->name});
    embedding_op_desc.SetInput("PadSeqLen",
                               {matched.at("pad_seq_len")->arg()->name});
    DirectedLink(matched.at("seq_lod"), matched.at("embedding"));
    DirectedLink(matched.at("pad_seq_len"), matched.at("embedding"));
    if (with_ln_) {
      DirectedLink(matched.at("layer_norm_out"), matched.at("xpu_encoder"));
      encoder_op_desc.SetInput("Input",
                               {matched.at("layer_norm_out")->arg()->name});
    } else {
      DirectedLink(matched.at("embedding_out"), matched.at("xpu_encoder"));
      encoder_op_desc.SetInput("Input",
                               {matched.at("embedding_out")->arg()->name});
    }
    embedding_instruct->ResetOp(embedding_op_desc, valid_places);
    matched.at("xpu_encoder")->stmt()->ResetOp(encoder_op_desc, valid_places);
  }

 private:
  bool with_ln_;
};

}  // namespace fusion

class XPUEmbeddingUnpadFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    std::vector<bool> with_lns{true, false};
    for (auto with_ln : with_lns) {
      fusion::XPUEmbeddingUnpadFuser fuser(with_ln);
      fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__embedding_unpad_fuse_pass,
                  paddle::lite::mir::XPUEmbeddingUnpadFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__embedding_with_eltwise_add");
