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

#pragma once
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * Mark the place of the variables in the SSAGrpah, it will inference the
 * variables' place by the kernels outputs them.
 */
class VariablePlaceInferencePass : public DebugPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

 private:
  // Mark the place of input arguments.
  void MarkInputPlace(SSAGraph* graph) {
    CHECK(!graph->inputs().empty()) << "graph's inputs should be set";
    for (const auto& v : graph->inputs()) {
      // the feed op might in the inputs
      if (v->IsStmt()) {
        VLOG(4) << "found kernel in inputs " << v->AsStmt().op_type();
        continue;
      }
    }
  }

  void CheckAllArgumentTypeDetermined(SSAGraph* graph) {
    for (auto& node : graph->mutable_nodes()) {
      if (node.IsArg()) {
        if (node.inlinks.size() == 0 && node.outlinks.size() == 0) {
          // empty node
          continue;
        }
        CHECK(node.AsArg().type) << "node " << node.AsArg().name
                                 << " type not determined, " << &node;
      }
    }
  }

  // Set the type of the weight
  void SetWeightType(Node* weight_node,
                     const LiteType& type,
                     const std::map<std::string, bool>& with_targets) {
    VLOG(4) << "type.precision():" << PrecisionRepr(type.precision());
    if (with_targets.at("kFPGA")) {
      weight_node->AsArg().type = LiteType::GetTensorTy(
          TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    } else if (with_targets.at("kOpenCL")) {
      weight_node->AsArg().type = LiteType::GetTensorTy(
          TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    } else if (with_targets.at("kCUDA")) {
      weight_node->AsArg().type = LiteType::GetTensorTy(
          TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    } else if (with_targets.at("kMetal") &&
               type.precision() == PRECISION(kUnk)) {
      weight_node->AsArg().type = LiteType::GetTensorTy(
          TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    } else if (with_targets.at("kXPU")) {
      weight_node->AsArg().type = LiteType::GetTensorTy(
          TARGET(kHost), PRECISION(kFloat), DATALAYOUT(kNCHW));
    } else {
      weight_node->AsArg().type = LiteType::GetTensorTy(
          TARGET(kHost), type.precision(), DATALAYOUT(kNCHW));
    }
  }

  // Update a's kUnk fields from b's fields.
  void UpdateTypeFrom(const Type** a, const Type* b) {
    auto target = (*a)->target();
    auto precision = (*a)->precision();
    auto layout = (*a)->layout();
    if (target == TARGET(kUnk)) {
      target = b->target();
    }
    if (precision == PRECISION(kUnk)) {
      precision = b->precision();
    }
    if (layout == DATALAYOUT(kUnk)) {
      layout = b->layout();
    }
    if ((*a)->IsTensor() && b->IsTensor()) {
      *a = LiteType::GetTensorTy(target, precision, layout);
    } else if ((*a)->IsTensor() && b->IsTensorList()) {
      *a = LiteType::GetTensorTy(target, precision, layout);
    } else if ((*a)->IsTensorList() && b->IsTensorList()) {
      *a = LiteType::GetTensorListTy(target, precision, layout);
    }
  }

  // Update tensors' precision according to kernel registry.
  // eg. Kernel con2d_fp16 is picked as the actual implementaion for conv2d op.
  //        con2d_fp16 kernel registry { input: X (host\kFloat16\NCHW)   output:
  //        Out (host\kFloat16\NCHW) }
  //        conv2d op_info: X:var1(precisionFloat32), Out:var2(precsionFloat32)
  //        after InferenceArgumentPlace is operated, related tensors will be
  //        updated:
  //        conv2d op_info: X:var1(precisionFloat16), Out:var2(precsionFloat16)

  void InferenceArgumentPlace(SSAGraph* graph) {
    auto& valid_places = graph->valid_places();
    auto valid_places_has_target = [&](TargetType t) -> bool {
      for (auto& p : valid_places) {
        if (p.target == t) {
          return true;
        }
      }
      return false;
    };
    std::map<std::string, bool> with_targets{
        {"kOpenCL", valid_places_has_target(TARGET(kOpenCL))},
        {"kCUDA", valid_places_has_target(TARGET(kCUDA))},
        {"kFPGA", valid_places_has_target(TARGET(kFPGA))},
        {"kMetal", valid_places_has_target(TARGET(kMetal))},
        {"kXPU", valid_places_has_target(TARGET(kXPU))},
    };
    VLOG(4) << "with_targets['kOpenCL']:" << with_targets["kOpenCL"];
    VLOG(4) << "with_targets['kFPGA']:" << with_targets["kFPGA"];
    VLOG(4) << "with_targets['kXPU']:" << with_targets["kXPU"];

    VLOG(3) << "param-type-registry:\n" << ParamTypeRegistry::Global();
    for (auto& node : graph->StmtTopologicalOrder()) {
      auto& inst = node->AsStmt();
      const auto* op_info = inst.op_info();
      const auto& op_type = op_info->Type();
      auto& kernel = inst.picked_kernel();

      // The IoCopyOp is a tool operator, it won't support the type inference.
      // in fpga, we has io_copy+cali+layout tool ops, so we need type inference
      // for tool operator
      if (with_targets["kFPGA"] || with_targets["kOpenCL"]) {
        VLOG(3) << "skip 'io_copy' if target is FPGA or OpenCL";
        if (op_type == "io_copy") continue;
      }

      // Infering the input and output variable's place according to the
      // declaration of I/O arguments of the picked kernel of the op
      VLOG(4) << "Op " << op_info->Repr();
      for (auto* in_node : node->inlinks) {
        auto& var = in_node->AsArg();
        const auto& var_name = var.name;
        auto* var_type = &var.type;
        std::string arg_name;

        CHECK(op_info->GetInputArgname(var_name, &arg_name))
            << "Can not find the input argument for var " << var_name;
        VLOG(4) << " - input arg name:" << arg_name << " var name:" << var_name;

        const auto* decl_type = kernel.GetInputDeclType(arg_name);
        CHECK(decl_type);
        if (!(*var_type)) {
          VLOG(4) << "set type " << *decl_type << " " << var_name;
          if (var.is_weight) {
            SetWeightType(in_node, *decl_type, with_targets);
          } else {
            *var_type = decl_type;
          }
        } else if (!(*var_type)->place().is_valid()) {
          if (var.is_weight && with_targets["kMetal"]) {
            SetWeightType(in_node, **var_type, with_targets);
          } else if (decl_type->precision() == PRECISION(kInt8) ||
                     (decl_type->precision() == PRECISION(kFP16) &&
                      decl_type->target() != TARGET(kOpenCL))) {
            *var_type = decl_type;
          } else {
            // If is quantization, infer the Int8 type.
            if (decl_type->precision() == PRECISION(kInt8)) {
              *var_type = decl_type;
            } else {
              UpdateTypeFrom(var_type, decl_type);
            }
          }
        }
      }
      for (auto* out_node : node->outlinks) {
        auto& var = out_node->AsArg();
        const auto& var_name = var.name;
        auto* var_type = &var.type;
        std::string arg_name;
        CHECK(op_info->GetOutputArgname(var_name, &arg_name))
            << "Can not find the output argument for var " << var_name;
        VLOG(4) << " - output arg name:" << arg_name
                << " var name:" << var_name;
        const auto* decl_type = kernel.GetOutputDeclType(arg_name);
        if (!(*var_type)) {
          VLOG(4) << "set type " << *decl_type << " " << var_name;
          *var_type = decl_type;
          // If one op's out is anothor op's weight, its is_weight or is_persist
          // attr should be false, otherwise an io_copy_once op may be insert.
          var.is_weight = false;
          var.is_persist = false;
        } else if (!(*var_type)->place().is_valid()) {
          // If is quantization, infer the Int8 type.
          if (decl_type->precision() == PRECISION(kInt8) ||
              (decl_type->precision() == PRECISION(kFP16) &&
               decl_type->target() != TARGET(kOpenCL))) {
            *var_type = decl_type;
          } else {
            UpdateTypeFrom(var_type, decl_type);
          }
        }
      }
    }
  }

  // For kernel whose input(X) and output(Out) are both defined as any
  // precision, while there is no dtype attribute from which we can determine
  // output(Out)'s precsion, we will update output(Out)'s precision directly
  // from input(X)'s precision.
  // eg.
  //     reshape kernel registry { input: X (host\kAny\NCHW)   output: Out
  //     (host\kAny\NCHW) }
  //     reshape op_info: X:var1(precisionFloat16), Out:var2(precsionFloat)
  //     after InferenceKernelWithUncertainPrecision is operated reshape op
  //     will be updated:
  //     reshape op_info: X:var1(precisionFloat16), Out:var2(precsionFloat16)
  void InferenceKernelWithUncertainPrecision(SSAGraph* graph) {
    std::vector<std::string> skiped_ops = {"feed",
                                           "fetch",
                                           "while",
                                           "subgraph",
                                           "io_copy",
                                           "io_copy_once",
                                           "cast"};
    for (auto& node : graph->StmtTopologicalOrder()) {
      auto& inst = node->AsStmt();
      const auto* op_info = inst.op_info();
      const auto& op_type = op_info->Type();
      auto& kernel = inst.picked_kernel();
      // Preprocessing for some special kernels
      if (InferQuantizedConcatOutputPrecision(node)) continue;
      if (InferQuantizedSubgraphOutputPrecision(node)) continue;
      if (std::find(skiped_ops.begin(), skiped_ops.end(), op_type) ==
              skiped_ops.end() &&
          op_info->HasInput("X") && op_info->HasOutput("Out") &&
          !op_info->HasAttr("dtype")) {
        const auto* decl_input_type = kernel.GetInputDeclType("X");
        const auto* decl_output_type = kernel.GetOutputDeclType("Out");
        if (decl_input_type->IsTensor() && decl_output_type->IsTensor() &&
            decl_input_type->precision() == PRECISION(kAny) &&
            decl_output_type->precision() == PRECISION(kAny)) {
          // update op's input variables precision from graph nodes info
          //    ps. op's input variables are stored in exec_scope, while
          //        graph node info is a temporary structure.
          auto UpdateOpInputsFromNodeInfo = [&]() {
            for (auto* in : node->inlinks) {
              if (!(in->AsArg().is_weight) && in->AsArg().type->IsTensor()) {
                auto in_arg_name = in->AsArg().name;
                auto* tmp_tensor = node->AsStmt()
                                       .op()
                                       ->scope()
                                       ->Var(in_arg_name)
                                       ->GetMutable<lite::Tensor>();
                tmp_tensor->set_precision(in->AsArg().type->precision());
              }
            }
          };

          // update graph nodes precision info from op's output variables
          //    ps. op's output variables are stored in exec_scope, while
          //        graph node info is a temporary structure.
          auto UpdateNodeInfoFromOpOutputs = [&] {
            for (auto* out : node->outlinks) {
              if (!(out->AsArg().is_weight) && out->AsArg().type->IsTensor()) {
                auto out_arg_name = out->AsArg().name;
                auto* tmp_tensor = node->AsStmt()
                                       .op()
                                       ->scope()
                                       ->Var(out_arg_name)
                                       ->GetMutable<lite::Tensor>();
                out->AsArg().type =
                    LiteType::GetTensorTy(out->AsArg().type->target(),
                                          tmp_tensor->precision(),
                                          out->AsArg().type->layout());
              }
            }
          };

          // update op's input variables precision from graph nodes info
          UpdateOpInputsFromNodeInfo();
          // update op's output precision from input precision by applying
          // InferType
          inst.op()->InferType();
          // update graph nodes precision info from op's output variables
          UpdateNodeInfoFromOpOutputs();
        }
      }
    }
  }

  // Only for the concat kernel whose output argument precision is defined as
  // PRECISION(kAny), force to set the output precision to PRECISION(kFloat) if
  // the precision of any input variable is PRECISION(kInt8) with the
  // quantizaiton parameters
  bool InferQuantizedConcatOutputPrecision(Node* op_node) {
    bool skip = false;
    auto& inst = op_node->AsStmt();
    const auto* op_info = inst.op_info();
    const auto& op_type = op_info->Type();
    auto& kernel = inst.picked_kernel();
    if (op_type != "concat") return false;
    const auto* decl_output_type = kernel.GetOutputDeclType("Out");
    if (decl_output_type->precision() != PRECISION(kAny)) return skip;
    for (auto* in_var_node : op_node->inlinks) {
      CHECK(in_var_node->IsArg());
      CHECK(in_var_node->AsArg().type);
      auto in_var_name = in_var_node->AsArg().name;
      auto in_var_type = in_var_node->AsArg().type;
      if (!op_info->HasInputScale(in_var_name)) continue;
      if (in_var_type->precision() != PRECISION(kInt8)) continue;
      // If the precision of any input variable is PRECISION(kInt8) with
      // quantization parameters, then force to set the output precision to
      // PRECISION(kFloat)
      CHECK_EQ(op_node->outlinks.size(), 1);
      auto out_var_node = op_node->outlinks.front();
      CHECK(out_var_node->IsArg());
      CHECK(out_var_node->AsArg().type);
      auto out_var_name = out_var_node->AsArg().name;
      auto& out_var_type = out_var_node->AsArg().type;
      if (in_var_type->IsTensor()) {
        out_var_type = LiteType::GetTensorTy(
            out_var_type->target(), PRECISION(kFloat), out_var_type->layout());
      } else if (in_var_type->IsTensorList()) {
        out_var_type = LiteType::GetTensorListTy(
            out_var_type->target(), PRECISION(kFloat), out_var_type->layout());
      }
      VLOG(4) << "Update " << out_var_name << " to " << *out_var_type;
      skip = true;
      break;
    }
    return skip;
  }

  // Only for the subgraph kernel whose output argument precision is defined as
  // PRECISION(kAny), infer the precision of the output data variables based on
  // the quantizaiton parameters
  bool InferQuantizedSubgraphOutputPrecision(Node* op_node) {
    bool skip = false;
    auto& inst = op_node->AsStmt();
    const auto* op_info = inst.op_info();
    const auto& op_type = op_info->Type();
    auto& kernel = inst.picked_kernel();
    if (op_type != "subgraph") return skip;
    if (kernel.target() != TARGET(kNNAdapter)) return skip;
    const auto* decl_output_type = kernel.GetOutputDeclType("Outputs");
    if (decl_output_type->precision() != PRECISION(kAny)) return skip;
    auto output_data_names =
        op_info->GetAttr<std::vector<std::string>>("output_data_names");
    for (auto* out_var_node : op_node->outlinks) {
      CHECK(out_var_node->IsArg());
      auto out_var_name = out_var_node->AsArg().name;
      // Only infer the precision of the output data variables which have the
      // quantization parameters
      if (!op_info->HasOutputScale(out_var_name)) continue;
      if (std::find(output_data_names.begin(),
                    output_data_names.end(),
                    out_var_name) == output_data_names.end())
        continue;
      CHECK(out_var_node->AsArg().type);
      auto& out_var_type = out_var_node->AsArg().type;
      auto out_var_target = out_var_type->target();
      auto out_var_precision = out_var_type->precision();
      // Skip if its precision is already set
      if (out_var_precision == PRECISION(kInt8)) continue;
      auto out_var_layout = out_var_type->layout();
      // Set the precision of the output variable to PRECISION(kInt8)
      if (out_var_type->IsTensor()) {
        out_var_type = LiteType::GetTensorTy(
            out_var_target, PRECISION(kInt8), out_var_layout);
      } else if (out_var_type->IsTensorList()) {
        out_var_type = LiteType::GetTensorListTy(
            out_var_target, PRECISION(kInt8), out_var_layout);
      }
      VLOG(4) << "Update " << out_var_name << " to " << *out_var_type;
      skip = true;
    }
    return skip;
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
