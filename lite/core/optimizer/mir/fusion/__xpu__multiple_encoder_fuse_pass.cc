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
#include <set>
#include <vector>
#include "lite/backends/xpu/math.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/xpu_pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {

namespace fusion {

class XPUMultipleEncoderFuser : public FuseBase {
 public:
  explicit XPUMultipleEncoderFuser(bool with_unpad = true,
                                   bool share_lod = true)
      : with_unpad_(with_unpad), share_lod_(share_lod) {}

  void BuildPattern() override {
    auto* encoder1_in = VarNode("encoder1_in")
                            ->assert_is_op_input("__xpu__encoder", "Input")
                            ->AsInput();
    auto* encoder1 = OpNode("encoder1", "__xpu__encoder");
    auto* encoder1_out = VarNode("encoder1_out")
                             ->assert_is_op_output("__xpu__encoder", "Output")
                             ->assert_only_one_output();
    auto* encoder2 = OpNode("encoder2", "__xpu__encoder");
    if (with_unpad_) {
      encoder1->assert_op_attr<bool>("adaptive_seqlen", true);
      encoder2->assert_op_attr<bool>("adaptive_seqlen", true);
      auto* seq_lod1 = VarNode("seq_lod_1")
                           ->assert_is_op_input("__xpu__encoder", "SeqLod")
                           ->AsInput();
      auto* pad_seq_len1 =
          VarNode("pad_seq_len1")
              ->assert_is_op_input("__xpu__encoder", "PadSeqLen")
              ->AsInput();

      encoder1_out->assert_is_op_input("sequence_unpad", "X");
      auto* seq_unpad = OpNode("seq_unpad", "sequence_unpad");
      auto* seq_unpad_out = VarNode("seq_unpad_out")
                                ->assert_is_op_output("sequence_unpad", "Out")
                                ->assert_is_op_input("__xpu__encoder", "Input")
                                ->assert_only_one_output();
      if (share_lod_) {
        encoder2->LinksFrom({seq_lod1, pad_seq_len1});
      } else {
        encoder1->assert_op_attr<bool>("token_pruning", true);
        auto* seq_lod2 = VarNode("seq_lod_2")
                             ->assert_is_op_input("__xpu__encoder", "SeqLod")
                             ->AsInput();
        auto* pad_seq_len2 =
            VarNode("pad_seq_len2")
                ->assert_is_op_input("__xpu__encoder", "PadSeqLen")
                ->AsInput();
        encoder2->LinksFrom({seq_lod2, pad_seq_len2});  // clsinds
      }
      *encoder1_in >> *encoder1 >> *encoder1_out >> *seq_unpad >>
          *seq_unpad_out >> *encoder2;
      encoder1->LinksFrom({seq_lod1, pad_seq_len1});
    } else {
      encoder1->assert_op_attr<bool>("adaptive_seqlen", false);
      encoder2->assert_op_attr<bool>("adaptive_seqlen", false);
      auto* mask = VarNode("mask")
                       ->assert_is_op_input("__xpu__encoder", "Mask")
                       ->AsInput();
      encoder1_out->assert_is_op_input("__xpu__encoder", "Input");
      *mask >> *encoder1;
      *mask >> *encoder2;
      *encoder1_in >> *encoder1 >> *encoder1_out >> *encoder2;
    }
    nodes2rm_.clear();
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    VLOG(3) << "______xpu_multiple_encoder_______";
    auto* encoder1_node = matched.at("encoder1");
    auto* encoder2_node = matched.at("encoder2");
    if (encoder1_node->stmt()->op_info()->HasAttr("DELETED") ||
        encoder2_node->stmt()->op_info()->HasAttr("DELETED")) {
      return;
    }
    if (with_unpad_) {
      nodes2rm_.insert(matched.at("seq_unpad"));
      nodes2rm_.insert(matched.at("seq_unpad_out"));
      if (share_lod_ && ShareSameAttrs(encoder1_node, encoder2_node)) {
        MergeEncoders(graph, matched, encoder1_node, encoder2_node);
      } else {
        encoder1_node->stmt()->mutable_op_info()->SetAttr<bool>("do_padding",
                                                                false);
        encoder2_node->stmt()->mutable_op_info()->SetInput(
            "Input", {matched.at("encoder1_out")->arg()->name});
        DirectedLink(matched.at("encoder1_out"), encoder2_node);
      }
    } else {
      if (ShareSameAttrs(encoder1_node, encoder2_node)) {
        MergeEncoders(graph, matched, encoder1_node, encoder2_node);
      }
    }
  }
  void DeleteNodes2rm(SSAGraph* graph) {
    GraphSafeRemoveNodes(graph, nodes2rm_);
  }

 private:
  bool with_unpad_;
  bool share_lod_;
  std::set<const Node*> nodes2rm_;

  // The functions below can only be used on encoders
  bool ShareSameAttrs(Node* op1, Node* op2) {
    cpp::OpDesc op1_desc = *op1->stmt()->mutable_op_info();
    cpp::OpDesc op2_desc = *op2->stmt()->mutable_op_info();
    std::vector<std::string> neither_bool_attrs = {"token_pruning"};
    std::vector<std::string> bool_attrs = {"enable_qkv_fusion", "norm_before"};
    std::vector<std::string> int_attrs = {
        "act_type", "hidden_dim", "head_num", "head_dim", "intermediate_size"};
    std::vector<std::string> float_attrs = {"alpha"};
    std::vector<std::string> string_vec_attrs = {"precision", "quant_type"};
    for (std::string& attr : neither_bool_attrs) {
      if (op1_desc.GetAttr<bool>(attr) || op2_desc.GetAttr<bool>(attr)) {
        return false;
      }
    }
    for (std::string& attr : bool_attrs) {
      if (op1_desc.GetAttr<bool>(attr) != op2_desc.GetAttr<bool>(attr)) {
        return false;
      }
    }
    for (std::string& attr : int_attrs) {
      if (op1_desc.GetAttr<int>(attr) != op2_desc.GetAttr<int>(attr)) {
        return false;
      }
    }
    for (std::string& attr : float_attrs) {
      if (std::abs(op1_desc.GetAttr<float>(attr) -
                   op2_desc.GetAttr<float>(attr)) > 1e-6) {
        return false;
      }
    }
    for (std::string& attr : string_vec_attrs) {
      if (!SameVecAttr<std::string>(op1_desc, op2_desc, attr)) {
        return false;
      }
    }
    return true;
  }

  template <typename T>
  bool SameVecAttr(const cpp::OpDesc& op1_desc,
                   const cpp::OpDesc& op2_desc,
                   const std::string& attr) {
    std::vector<T> attr1 = op1_desc.GetAttr<std::vector<T>>(attr);
    std::vector<T> attr2 = op2_desc.GetAttr<std::vector<T>>(attr);
    if (attr1.size() != attr2.size()) {
      return false;
    }
    for (int i = 0; i < attr1.size(); ++i) {
      if (attr1[i] != attr2[i]) {
        return false;
      }
    }
    return true;
  }

  bool isLinkTo(Node* node1, Node* node2) {
    for (auto* node1_out : node1->outlinks) {
      if (node1_out == node2) {
        return true;
      }
    }
    return false;
  }

  void LinkInputNodes(Node* op1, Node* op2, Node* new_op_node) {
    for (auto* op1_in : op1->inlinks) {
      DirectedLink(op1_in, new_op_node);
    }
    for (auto* op2_in : op2->inlinks) {
      if (!isLinkTo(op1, op2_in) && !isLinkTo(op2_in, op1)) {
        DirectedLink(op2_in, new_op_node);
      }
    }
    for (auto* op2_out : op2->outlinks) {
      DirectedLink(new_op_node, op2_out);
    }
  }

  void MergeParams(const key2nodes_t& matched,
                   Node* encoder1_node,
                   Node* encoder2_node,
                   Node* new_encoder_node) {
    auto* new_encoder_instruct = new_encoder_node->stmt();
    auto new_encoder_op_desc = *encoder2_node->stmt()->op_info();
    auto new_encoder_op = new_encoder_instruct->op();
    auto encoder1_op_desc = *encoder1_node->stmt()->op_info();
    auto encoder2_op_desc = *encoder2_node->stmt()->op_info();

    int layers = encoder1_op_desc.GetAttr<int>("n_layers") +
                 encoder2_op_desc.GetAttr<int>("n_layers");
    new_encoder_op_desc.SetAttr<int>("n_layers", layers);
    std::vector<std::string> Attrs = {"weight_max", "io_max"};
    for (std::string& attr_name : Attrs) {
      std::vector<float> attr1 =
          encoder1_op_desc.GetAttr<std::vector<float>>(attr_name);
      std::vector<float> attr2 =
          encoder2_op_desc.GetAttr<std::vector<float>>(attr_name);
      attr2.insert(attr2.begin(), attr1.begin(), attr1.end());
      new_encoder_op_desc.SetAttr<std::vector<float>>(attr_name, attr2);
    }
    std::vector<std::string> Inputs = {
        "FCWeight", "FCBias", "LNScale", "LNBias"};
    for (std::string& input : Inputs) {
      std::vector<std::string> tensor_names1 = encoder1_op_desc.Input(input);
      std::vector<std::string> tensor_names2 = encoder2_op_desc.Input(input);
      tensor_names2.insert(
          tensor_names2.begin(), tensor_names1.begin(), tensor_names1.end());
      new_encoder_op_desc.SetInput(input, tensor_names2);
    }
    new_encoder_op_desc.SetInput("Input", encoder1_op_desc.Input("Input"));
    new_encoder_instruct->ResetOp(new_encoder_op_desc,
                                  new_encoder_op->valid_places());
  }

  void MergeEncoders(SSAGraph* graph,
                     const key2nodes_t& matched,
                     Node* encoder1_node,
                     Node* encoder2_node) {
    auto encoder_op = LiteOpRegistry::Global().Create("__xpu__encoder");
    encoder_op->Attach(*encoder2_node->stmt()->op_info(),
                       encoder2_node->stmt()->op()->scope());
    Node* new_encoder_node = graph->GraphCreateInstructNode(
        encoder_op, encoder2_node->stmt()->op()->valid_places());
    LinkInputNodes(
        encoder1_node,
        encoder2_node,
        new_encoder_node);  // link input nodes of encoder1 to encoder2
    MergeParams(matched, encoder1_node, encoder2_node, new_encoder_node);
    nodes2rm_.insert(matched.at("encoder1"));
    nodes2rm_.insert(matched.at("encoder1_out"));
    nodes2rm_.insert(matched.at("encoder2"));
    encoder1_node->stmt()->mutable_op_info()->SetAttr<bool>("DELETED", true);
    encoder2_node->stmt()->mutable_op_info()->SetAttr<bool>("DELETED", true);
  }
};

}  // namespace fusion

class XPUMultipleEncoderFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    std::vector<bool> with_unpads{true, false};
    std::vector<bool> share_lods{true, false};
    for (auto with_unpad : with_unpads) {
      for (auto share_lod : share_lods) {
        fusion::XPUMultipleEncoderFuser fuser(with_unpad, share_lod);
        size_t encoder_num = fuser(graph.get());
        fuser.DeleteNodes2rm(graph.get());
        if (encoder_num > 0) {
          VLOG(3) << "Found " << encoder_num + 1 << " encoders.";
        }
        for (int i = 0; i < encoder_num; ++i) {
          VLOG(3) << "run" << i << "th encoder passes.";
          fusion::XPUMultipleEncoderFuser fuser2(with_unpad, share_lod);
          fuser2(graph.get());
          fuser2.DeleteNodes2rm(graph.get());
        }
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__multiple_encoder_fuse_pass,
                  paddle::lite::mir::XPUMultipleEncoderFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__encoder");
