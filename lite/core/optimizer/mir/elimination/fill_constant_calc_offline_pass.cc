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

#include "lite/core/optimizer/mir/elimination/fill_constant_calc_offline_pass.h"
#include <algorithm>
#include <cmath>
#include <list>
#include <memory>
#include <set>
#include <vector>
#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"
#include "lite/core/optimizer/mir/pattern_matcher.h"
#include "lite/core/optimizer/mir/ssa_graph_utils.h"
#include "lite/model_parser/cpp_desc.h"

namespace paddle {
namespace lite {
namespace mir {

template <typename T>
void FillConstData(lite::Tensor* out_t, T value) {
  auto output_data = out_t->mutable_data<T>();
  for (int i = 0; i < out_t->numel(); i++) {
    output_data[i] = value;
  }
}

void FillConstantCalcOfflinePass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  RemoveFillConstantPattern(graph);
}

void FillConstantCalcOfflinePass::RemoveFillConstantPattern(
    const std::unique_ptr<SSAGraph>& graph) {
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->AsStmt().op_type() != "fill_constant") continue;
    auto outlinks = node->outlinks;
    bool has_extra_producers = false;
    for (auto& out_link : outlinks) {
      if (HasExtraProducers(
              graph.get(), out_link->arg()->name, {"fill_constant"})) {
        has_extra_producers = true;
        break;
      }
    }
    if (has_extra_producers) {
      VLOG(5) << "WARNING: Unsupported for op output var containing multiple "
                 "producers";
      continue;
    }
    std::set<const Node*> nodes2rm_;
    auto& fill_constant_instruct = node->AsStmt();
    auto* scope = fill_constant_instruct.op()->scope();
    auto op_desc = fill_constant_instruct.mutable_op_info();
    if ((op_desc->HasInput("ValueTensor") &&
         !op_desc->Input("ValueTensor").empty()) ||
        (op_desc->HasInput("str_value") &&
         !op_desc->GetAttr<std::string>("str_value").empty())) {
      VLOG(5) << "WARNING: Unsupported ValueTensor input or str_value input "
                 "for fill_contant op.";
      continue;
    } else if (!op_desc->HasAttr("value")) {
      VLOG(5) << "WARNING: One of ValueTensor, str_value(attr) or value(attr) "
                 "must be set.";
      continue;
    }
    if ((op_desc->HasInput("ShapeTensor") &&
         !op_desc->Input("ShapeTensor").empty()) ||
        (op_desc->HasInput("ShapeTensorList") &&
         !op_desc->Input("ShapeTensorList").empty())) {
      VLOG(5) << "WARNING: Unsupported ShapeTensor or ShapeTensorList input "
                 "for fill_contant op.";
      continue;
    } else if (!op_desc->HasAttr("shape")) {
      VLOG(5) << "WARNING: One of ShapeTensor, ShapeTensorList or shape(attr) "
                 "must be set.";
      continue;
    }
    // Get fill_constant's attr
    auto dtype = op_desc->GetAttr<int>("dtype");
    std::string str_value{};
    if (op_desc->HasAttr("str_value")) {
      str_value = op_desc->GetAttr<std::string>("str_value");
    }

    float value;
    if (str_value.empty()) {
      value = op_desc->GetAttr<float>("value");
    } else {
      // handle NaN/Inf first, which cannot be read from stream.
      if (str_value == "inf") {
        value = std::numeric_limits<float>::quiet_NaN();
      } else if (str_value == "-inf") {
        value = std::numeric_limits<float>::quiet_NaN();
      } else if (str_value == "nan") {
        value = std::numeric_limits<float>::quiet_NaN();
      } else {
        std::stringstream convert_stream(str_value);
        convert_stream >> value;
      }
    }
    std::vector<int64_t> shape;
    if (op_desc->GetAttrType("shape") ==
        OpAttrType::INTS) {  // paddle1.0 shape type is ints
      auto tmp_shape = op_desc->GetAttr<std::vector<int32_t>>("shape");
      shape.resize(tmp_shape.size());
      shape.assign(tmp_shape.begin(), tmp_shape.end());
    } else {
      shape = op_desc->GetAttr<std::vector<int64_t>>("shape");
    }

    // Get fill_constant's output tensor
    auto out_var = scope->FindVar(op_desc->Output("Out").front());
    auto out_t = out_var->GetMutable<lite::Tensor>();
    out_t->Resize(DDim(shape));
    switch (dtype) {
      case static_cast<int>(lite::core::FluidType::BOOL):
        FillConstData<bool>(out_t, static_cast<bool>(value));
        break;
      case static_cast<int>(lite::core::FluidType::INT32):
        FillConstData<int32_t>(out_t, static_cast<int32_t>(value));
        break;
      case static_cast<int>(lite::core::FluidType::INT64):
        FillConstData<int64_t>(out_t, static_cast<int64_t>(value));
        break;
      case static_cast<int>(lite::core::FluidType::FP32):
        FillConstData<float>(out_t, static_cast<float>(value));
        break;
      default:
        VLOG(5) << "WARNING: Unsupported dtype for fill_constant op: " << dtype;
        continue;
    }
    // Offline calc fill_constant, only retain output tensor as persistable
    // tensor
    out_t->set_persistable(true);
    auto fill_constant_outlinks = node->outlinks;
    for (auto& fill_constant_out_link : fill_constant_outlinks) {
      fill_constant_out_link->arg()->is_weight = true;
    }
    nodes2rm_.insert(node);
    GraphSafeRemoveNodes(graph.get(), nodes2rm_);
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(fill_constant_calc_offline_pass,
                  paddle::lite::mir::FillConstantCalcOfflinePass)
    .BindTargets({TARGET(kNNAdapter), TARGET(kARM), TARGET(kOpenCL)});
