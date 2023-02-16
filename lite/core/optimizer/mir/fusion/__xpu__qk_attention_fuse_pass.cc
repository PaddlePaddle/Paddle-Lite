// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
// // //
// // // Licensed under the Apache License, Version 2.0 (the "License");
// // // you may not use this file except in compliance with the License.
// // // You may obtain a copy of the License at
// // //
// // //     http://www.apache.org/licenses/LICENSE-2.0
// // //
// // // Unless required by applicable law or agreed to in writing, software
// // // distributed under the License is distributed on an "AS IS" BASIS,
// // // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied.
// // // See the License for the specific language governing permissions and
// // // limitations under the License.

#include <memory>
#include <string>
#include "lite/backends/xpu/math.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher_high_api.h"

namespace paddle {
namespace lite {
namespace mir {
namespace fusion {

class XPUQkAttentionFuser : public FuseBase {
 public:
  explicit XPUQkAttentionFuser(const std::string& matmul_type = "matmul",
                               bool with_q_scale = true,
                               bool with_mask = true)
      : matmul_type_(matmul_type),
        with_q_scale_(with_q_scale),
        with_mask_(with_mask) {}

  void BuildPattern() override {
    auto* q = VarNode("q")->assert_is_op_input("reshape2", "X")->AsInput();
    auto* q_reshape2 = OpNode("q_reshape2", "reshape2")->AsIntermediate();
    // TODO(TingShen): check shape
    auto* q_reshape2_out = VarNode("q_reshape2_out")
                               ->assert_is_op_output("reshape2", "Out")
                               ->assert_is_op_input("transpose2", "X")
                               ->AsIntermediate();
    auto* q_reshape2_xshape = VarNode("q_reshape2_xshape")
                                  ->assert_is_op_output("reshape2", "XShape")
                                  ->AsIntermediate();
    std::string target_op_type = matmul_type_;
    if (with_q_scale_) {
      target_op_type = "scale";
    }
    auto* q_transpose2 = OpNode("q_transpose2", "transpose2")->AsIntermediate();
    auto* q_transpose2_out = VarNode("q_transpose2_out")
                                 ->assert_is_op_output("transpose2", "Out")
                                 ->assert_is_op_input(target_op_type, "X")
                                 ->AsIntermediate();
    auto* q_transpose2_xshape =
        VarNode("q_transpose2_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();

    PMNode* q_scale = nullptr;
    PMNode* q_scale_out = nullptr;
    if (with_q_scale_) {
      q_scale = OpNode("q_scale", "scale")->AsIntermediate();
      q_scale_out = VarNode("q_scale_out")
                        ->assert_is_op_output("scale", "Out")
                        ->assert_is_op_input(matmul_type_, "X")
                        ->AsIntermediate();
    }
    auto* k = VarNode("k")->assert_is_op_input("reshape2", "X")->AsInput();
    auto* k_reshape2 = OpNode("k_reshape2", "reshape2")->AsIntermediate();
    auto* k_reshape2_out = VarNode("k_reshape2_out")
                               ->assert_is_op_output("reshape2", "Out")
                               ->assert_is_op_input("transpose2", "X")
                               ->AsIntermediate();
    auto* k_reshape2_xshape = VarNode("k_reshape2_xshape")
                                  ->assert_is_op_output("reshape2", "XShape")
                                  ->AsIntermediate();
    auto* k_transpose2 = OpNode("k_transpose2", "transpose2")->AsIntermediate();
    auto* k_transpose2_out = VarNode("k_transpose2_out")
                                 ->assert_is_op_output("transpose2", "Out")
                                 ->assert_is_op_input(matmul_type_, "Y")
                                 ->AsIntermediate();
    auto* k_transpose2_xshape =
        VarNode("k_transpose2_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();

    auto* qk_matmul = OpNode("qk_matmul", matmul_type_)->AsIntermediate();
    auto* qk_matmul_out = VarNode("qk_matmul_out")
                              ->assert_is_op_output(matmul_type_, "Out")
                              ->AsIntermediate();
    PMNode* qk_mask = nullptr;
    PMNode* qk_add = nullptr;
    PMNode* qk_add_out = nullptr;
    if (with_mask_) {
      qk_matmul_out->assert_is_op_input("elementwise_add", "X");
      qk_mask = VarNode("qk_mask")
                    ->assert_is_op_input("elementwise_add", "Y")
                    ->AsInput();
      qk_add = OpNode("qk_add", "elementwise_add")->AsIntermediate();
      qk_add_out = VarNode("qk_add_out")
                       ->assert_is_op_output("elementwise_add", "Out")
                       ->assert_is_op_input("softmax", "X")
                       ->AsIntermediate();
    } else {
      qk_matmul_out->assert_is_op_input("softmax", "X");
    }
    auto* qk_softmax = OpNode("qk_softmax", "softmax")->AsIntermediate();
    auto* qk_softmax_out = VarNode("qk_softmax_out")
                               ->assert_is_op_output("softmax", "Out")
                               ->AsOutput();

    *q >> *q_reshape2 >> *q_reshape2_out >> *q_transpose2 >> *q_transpose2_out;
    *q_reshape2 >> *q_reshape2_xshape;
    *q_transpose2 >> *q_transpose2_xshape;
    if (with_q_scale_) {
      *q_transpose2_out >> *q_scale >> *q_scale_out >> *qk_matmul;
    } else {
      *q_transpose2_out >> *qk_matmul;
    }
    *k >> *k_reshape2 >> *k_reshape2_out >> *k_transpose2 >>
        *k_transpose2_out >> *qk_matmul >> *qk_matmul_out;
    *k_reshape2 >> *k_reshape2_xshape;
    *k_transpose2 >> *k_transpose2_xshape;
    if (with_mask_) {
      *qk_mask >> *qk_add;
      *qk_matmul_out >> *qk_add >> *qk_add_out >> *qk_softmax >>
          *qk_softmax_out;
    } else {
      *qk_matmul_out >> *qk_softmax >> *qk_softmax_out;
    }
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    VLOG(3) << "______xpu_qk_attention_____";
    float scale_val = 0.f;
    if (with_q_scale_) {
      scale_val =
          matched.at("q_scale")->stmt()->op_info()->GetAttr<float>("scale");
    } else {
      scale_val =
          matched.at("qk_matmul")->stmt()->op_info()->GetAttr<float>("alpha");
    }
    std::vector<int> shape = matched.at("q_reshape2")
                                 ->stmt()
                                 ->op_info()
                                 ->GetAttr<std::vector<int>>("shape");
    int head_num = shape[2];
    int head_dim = shape[3];
    auto* scope = matched.at("qk_matmul")->stmt()->op()->scope();
    auto valid_places = matched.at("qk_matmul")->stmt()->op()->valid_places();
    cpp::OpDesc op_desc;
    op_desc.mutable_inputs()->clear();
    op_desc.mutable_outputs()->clear();
    op_desc.SetType("__xpu__qk_attention");
    op_desc.SetAttr<float>("alpha", scale_val);
    op_desc.SetAttr<int>("head_num", head_num);
    op_desc.SetAttr<int>("head_dim", head_dim);
    op_desc.SetInput("q", {matched.at("q")->arg()->name});
    op_desc.SetInput("k", {matched.at("k")->arg()->name});
    op_desc.SetOutput("output", {matched.at("qk_softmax_out")->arg()->name});
    if (with_mask_) {
      op_desc.SetInput("mask", {matched.at("qk_mask")->arg()->name});
    }

    // Set quant attributes
    int bit_len = 32;
    std::string precision;
    std::string quant_type;
    bool quant =
        matched.at("qk_matmul")->stmt()->op_info()->HasAttr("enable_quant") &&
        matched.at("qk_matmul")
            ->stmt()
            ->op_info()
            ->GetAttr<bool>("enable_quant");
    if (quant) {
      bit_len = matched.at("qk_matmul")
                    ->stmt()
                    ->op_info()
                    ->GetAttr<int>("bit_length");
      if (bit_len == 8) {
        precision = "int8";
      } else if (bit_len == 16) {
        precision = "int16";
      } else {
        LOG(FATAL) << "Unsupported quant bit length: " << bit_len;
      }
    }

    // TODO(TingShen): Remove encable_int16 & enable_int8.
    //                Use enable_quant and bit_length instead.
    if (matched.at("qk_matmul")->stmt()->op_info()->HasAttr("enable_int16") &&
        matched.at("qk_matmul")
            ->stmt()
            ->op_info()
            ->GetAttr<bool>("enable_int16")) {
      op_desc.SetAttr<bool>("enable_int16", true);
      quant = true;
      bit_len = 16;
      precision = "int16";
    }
    if (matched.at("qk_matmul")->stmt()->op_info()->HasAttr("enable_int8") &&
        matched.at("qk_matmul")
            ->stmt()
            ->op_info()
            ->GetAttr<bool>("enable_int8")) {
      op_desc.SetAttr<bool>("enable_int8", true);
      quant = true;
      bit_len = 8;
      precision = "int8";
    }
    if (quant) {
      quant_type = "per_tensor";
      op_desc.SetAttr<std::vector<float>>(
          "input_scale",
          {matched.at("q_reshape2")
               ->stmt()
               ->op_info()
               ->GetAttr<float>("out_threshold"),
           matched.at("k_reshape2")
               ->stmt()
               ->op_info()
               ->GetAttr<float>("out_threshold")});
      op_desc.SetAttr<std::vector<float>>(
          "output_scale",
          {matched.at("qk_softmax")
               ->stmt()
               ->op_info()
               ->GetAttr<float>("out_threshold")});
    } else {
      precision = "int31";
      quant_type = "no_quant";
#ifdef LITE_WITH_XPU
      /* To suppress linkage error, we use #ifdef here.*/
      /* TODO(TingShen): Add a global precision parameter. For compatibility, we
         now temporarily use multi_encoder_precision as the global precision
         setup
         for matrix multiplication ops.*/
      if (GetStringFromEnv("XPU_ENCODER_PRECISION", "int16") == "int31" ||
          lite::TargetWrapperXPU::xpu_runtime_ptr->multi_encoder_precision ==
              "int31") {
        precision = "int31";
      } else if (GetStringFromEnv("XPU_ENCODER_PRECISION", "int16") == "int8" ||
                 lite::TargetWrapperXPU::xpu_runtime_ptr
                         ->multi_encoder_precision == "int8") {
        precision = "int8";
      } else if (GetStringFromEnv("XPU_ENCODER_PRECISION", "int16") ==
                     "local_quant" ||
                 lite::TargetWrapperXPU::xpu_runtime_ptr
                         ->multi_encoder_precision == "local_quant") {
        precision = "local_quant";
        quant_type = "local_quant";
      } else {
        precision = "int16";
      }
#endif
    }
    op_desc.SetAttr<std::string>("precision", precision);
    op_desc.SetAttr<std::string>("quant_type", quant_type);

    auto op = LiteOpRegistry::Global().Create("__xpu__qk_attention");
    op->Attach(op_desc, scope);
    op->SetValidPlaces(valid_places);
    auto* op_node = graph->GraphCreateInstructNode(op, valid_places);

    DirectedLink(matched.at("q"), op_node);
    DirectedLink(matched.at("k"), op_node);
    if (with_mask_) {
      DirectedLink(matched.at("qk_mask"), op_node);
    }
    DirectedLink(op_node, matched.at("qk_softmax_out"));
  }

 private:
  std::string matmul_type_;
  bool with_q_scale_;
  bool with_mask_;
};

}  // namespace fusion

class XPUQkAttentionFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    std::vector<std::string> matmul_types{"matmul", "matmul_v2"};
    std::vector<bool> with_q_scales{true, false};
    std::vector<bool> with_masks{true, false};
    for (auto& matmul_type : matmul_types) {
      for (auto with_q_scale : with_q_scales) {
        for (auto with_mask : with_masks) {
          fusion::XPUQkAttentionFuser fuser(
              matmul_type, with_q_scale, with_mask);
          fuser(graph.get());
        }
      }
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__qk_attention_fuse_pass,
                  paddle::lite::mir::XPUQkAttentionFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__qk_attention");
