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

class XPUQkVAttentionFuser : public FuseBase {
 public:
  explicit XPUQkVAttentionFuser(const std::string& matmul_type = "matmul")
      : matmul_type_(matmul_type) {}

  void BuildPattern() override {
    auto* qk = VarNode("qk")->assert_is_op_input(matmul_type_, "X")->AsInput();
    auto* v = VarNode("v")->assert_is_op_input("reshape2", "X")->AsInput();
    auto* v_reshape2 = OpNode("v_reshape2", "reshape2")->AsIntermediate();
    auto* v_reshape2_out = VarNode("v_reshape2_out")
                               ->assert_is_op_output("reshape2", "Out")
                               ->assert_is_op_input("transpose2", "X")
                               ->AsIntermediate();
    auto* v_reshape2_xshape = VarNode("v_reshape2_xshape")
                                  ->assert_is_op_output("reshape2", "XShape")
                                  ->AsIntermediate();
    auto* v_transpose2 = OpNode("v_transpose2", "transpose2")->AsIntermediate();
    auto* v_transpose2_out = VarNode("v_transpose2_out")
                                 ->assert_is_op_output("transpose2", "Out")
                                 ->assert_is_op_input(matmul_type_, "Y")
                                 ->AsIntermediate();
    auto* v_transpose2_xshape =
        VarNode("v_transpose2_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();

    auto* qkv_matmul = OpNode("qkv_matmul", matmul_type_)->AsIntermediate();
    auto* qkv_matmul_out = VarNode("qkv_matmul_out")
                               ->assert_is_op_output(matmul_type_, "Out")
                               ->assert_is_op_input("transpose2", "X")
                               ->AsIntermediate();
    auto* qkv_transpose2 =
        OpNode("qkv_transpose2", "transpose2")->AsIntermediate();
    auto* qkv_transpose2_out = VarNode("qkv_transpose2_out")
                                   ->assert_is_op_output("transpose2", "Out")
                                   ->assert_is_op_input("reshape2", "X")
                                   ->AsIntermediate();
    auto* qkv_transpose2_xshape =
        VarNode("qkv_transpose2_xshape")
            ->assert_is_op_output("transpose2", "XShape")
            ->AsIntermediate();
    auto* qkv_reshape2 = OpNode("qkv_reshape2", "reshape2")->AsIntermediate();
    auto* qkv_reshape2_xshape = VarNode("qkv_reshape2_xshape")
                                    ->assert_is_op_output("reshape2", "XShape")
                                    ->AsIntermediate();
    auto* qkv_reshape2_out = VarNode("qkv_reshape2_out")
                                 ->assert_is_op_output("reshape2", "Out")
                                 ->AsOutput();
    *v >> *v_reshape2 >> *v_reshape2_out >> *v_transpose2 >>
        *v_transpose2_out >> *qkv_matmul;
    *qk >> *qkv_matmul;
    *v_reshape2 >> *v_reshape2_xshape;
    *v_transpose2 >> *v_transpose2_xshape;

    *qkv_matmul >> *qkv_matmul_out >> *qkv_transpose2 >> *qkv_transpose2_out >>
        *qkv_reshape2 >> *qkv_reshape2_out;
    *qkv_transpose2 >> *qkv_transpose2_xshape;
    *qkv_reshape2 >> *qkv_reshape2_xshape;
  }

  void InsertNewNode(SSAGraph* graph, const key2nodes_t& matched) override {
    VLOG(3) << "______xpu_qk_v_attention_____";
    std::vector<int> shape = matched.at("v_reshape2")
                                 ->stmt()
                                 ->op_info()
                                 ->GetAttr<std::vector<int>>("shape");
    int head_num = shape[2];
    int head_dim = shape[3];
    auto* scope = matched.at("qkv_matmul")->stmt()->op()->scope();
    auto valid_places = matched.at("qkv_matmul")->stmt()->op()->valid_places();
    cpp::OpDesc op_desc;
    op_desc.SetType("__xpu__qk_v_attention");
    op_desc.SetAttr<int>("head_num", head_num);
    op_desc.SetAttr<int>("head_dim", head_dim);
    op_desc.SetInput("qk", {matched.at("qk")->arg()->name});
    op_desc.SetInput("v", {matched.at("v")->arg()->name});
    op_desc.SetOutput("output", {matched.at("qkv_reshape2_out")->arg()->name});

    // Set quant attributes
    int bit_len = 32;
    std::string precision;
    std::string quant_type;
    bool quant =
        matched.at("qkv_matmul")->stmt()->op_info()->HasAttr("enable_quant") &&
        matched.at("qkv_matmul")
            ->stmt()
            ->op_info()
            ->GetAttr<bool>("enable_quant");
    if (quant) {
      bit_len = matched.at("qkv_matmul")
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
    if (matched.at("qkv_matmul")->stmt()->op_info()->HasAttr("enable_int16") &&
        matched.at("qkv_matmul")
            ->stmt()
            ->op_info()
            ->GetAttr<bool>("enable_int16")) {
      op_desc.SetAttr<bool>("enable_int16", true);
      quant = true;
      bit_len = 16;
      precision = "int16";
    }
    if (matched.at("qkv_matmul")->stmt()->op_info()->HasAttr("enable_int8") &&
        matched.at("qkv_matmul")
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
      std::vector<std::string> inputs = {"qk", "v_transpose2_out"};
      std::vector<float> input_scale;
      for (std::string& input : inputs) {
        input_scale.push_back(
            ((1 << bit_len) - 1) *
            matched.at("qkv_matmul")
                ->stmt()
                ->op_info()
                ->GetInputScale(matched.at(input)->arg()->name)[0]);
      }
      op_desc.SetAttr<std::vector<float>>("input_scale", input_scale);
      if (matched.at("qkv_matmul")
              ->stmt()
              ->op_info()
              ->HasAttr("out_threshold")) {
        op_desc.SetAttr<std::vector<float>>(
            "output_scale",
            {matched.at("qkv_matmul")
                 ->stmt()
                 ->op_info()
                 ->GetAttr<float>("out_threshold")});
      }
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

    auto op = LiteOpRegistry::Global().Create("__xpu__qk_v_attention");
    op->Attach(op_desc, scope);
    op->SetValidPlaces(valid_places);
    auto* op_node = graph->GraphCreateInstructNode(op, valid_places);

    DirectedLink(matched.at("qk"), op_node);
    DirectedLink(matched.at("v"), op_node);
    DirectedLink(op_node, matched.at("qkv_reshape2_out"));
  }

 private:
  std::string matmul_type_;
};

}  // namespace fusion

class XPUQkVAttentionFusePass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override {
    std::vector<std::string> matmul_types{"matmul", "matmul_v2"};
    for (auto& matmul_type : matmul_types) {
      fusion::XPUQkVAttentionFuser fuser(matmul_type);
      fuser(graph.get());
    }
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__qk_v_attention_fuse_pass,
                  paddle::lite::mir::XPUQkVAttentionFusePass)
    .BindTargets({TARGET(kXPU)})
    .BindKernel("__xpu__qk_v_attention");
