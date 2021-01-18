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
#include "lite/core/mir/pass.h"
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
    } else if ((*a)->IsTensorList() && b->IsTensorList()) {
      *a = LiteType::GetTensorListTy(target, precision, layout);
    }
  }

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
        {"kFPGA", valid_places_has_target(TARGET(kFPGA))}};
    VLOG(4) << "with_targets['kOpenCL']:" << with_targets["kOpenCL"];
    VLOG(4) << "with_targets['kFPGA']:" << with_targets["kFPGA"];

    VLOG(3) << "param-type-registry:\n" << ParamTypeRegistry::Global();
    for (auto& node : graph->StmtTopologicalOrder()) {
      auto& inst = node->AsStmt();
      const auto* op_info = inst.op_info();
      const auto& op_type = op_info->Type();
      auto& kernel = inst.picked_kernel();

      // The IoCopyOp is a tool operator, it won't support the type inference.
      // in fpga, we has io_copy+cali+layout tool ops, so we need type inference
      // for tool operator
      if ((!with_targets["kFPGA"]) && (!with_targets["kOpenCL"])) {
        VLOG(3) << "skip 'io_copy' if target is FPGA and OpenCL";
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
        if (!(*var_type)) {
          VLOG(4) << "set type " << *decl_type << " " << var_name;
          if (var.is_weight) {
            SetWeightType(in_node, *decl_type, with_targets);
          } else {
            *var_type = decl_type;
          }
        } else if (!(*var_type)->place().is_valid()) {
          // If is quantization, infer the Int8 type.
          if (decl_type->precision() == PRECISION(kInt8)) {
            *var_type = decl_type;
          } else {
            UpdateTypeFrom(var_type, decl_type);
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
          if (var.is_weight) {
            SetWeightType(out_node, *decl_type, with_targets);
          } else {
            *var_type = decl_type;
          }
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

 private:
  // The default target for arguments, e.g. load weights to CPU memory for
  // CUDA computation by default.
  TargetType argument_default_target_{TARGET(kHost)};
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
