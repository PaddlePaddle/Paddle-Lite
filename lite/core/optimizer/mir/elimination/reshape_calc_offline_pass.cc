// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/core/optimizer/mir/elimination/reshape_calc_offline_pass.h"
#include <algorithm>
#include <cmath>
#include <list>
#include <set>
#include "lite/core/optimizer/mir/pattern_matcher.h"
#include "lite/core/optimizer/mir/ssa_graph_utils.h"
#include "lite/model_parser/cpp_desc.h"

namespace paddle {
namespace lite {
namespace mir {

static bool CheckPositive(const DDim& dims) {
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] <= 0) {
      return false;
    }
  }
  return true;
}

std::vector<int64_t> ValidateShape(const std::vector<int>& shape,
                                   const DDim& input_dims) {
  const int64_t input_size = input_dims.production();

  // only one dimension can be set to -1, whose size will be automatically
  // infered.
  const int unk_dim_val = -1;
  const int copy_dim_val = 0;

  std::vector<int64_t> output_dims(shape.size());
  int64_t capacity = 1;
  int unk_dim_idx = -1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == unk_dim_val) {
      CHECK_EQ(unk_dim_idx, -1)
          << "Only one input dimension of Attr(shape) can be unknown.";
      unk_dim_idx = i;
    } else if (shape[i] == copy_dim_val) {
      CHECK_LT(i, input_dims.size())
          << "The index of dimension to copy from input shape must be less "
             "than the size of input shape.";
    } else {
      CHECK_GT(shape[i], 0) << "Each input dimension of Attr(shape) must not "
                               "be negtive except one unknown dimension.";
    }

    int64_t output_dim_i =
        shape[i] ? static_cast<int64_t>(shape[i]) : input_dims[i];
    output_dims[i] = output_dim_i;
    capacity *= output_dim_i;
  }

  if (unk_dim_idx != -1) {
    if (CheckPositive(input_dims)) {
      // input_size < 0 and is un-determinate in compile time, skip the check,
      // for example, input_dims = [-1, 8, 1, 1], shape = [-1, 3, 8],
      // capacity = -24, input_size = -8, output_shape[0] = 0
      // the following check will fail.
      output_dims[unk_dim_idx] = -input_size / capacity;
      CHECK_EQ(output_dims[unk_dim_idx] * capacity, -input_size)
          << "Invalid shape is given.";
    } else {
      output_dims[unk_dim_idx] = -1;
    }
  } else {
    CHECK_EQ(capacity, input_size) << "Invalid shape is given.";
  }
  return output_dims;
}

void ReshapeCalcOfflinePass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  RemoveReshapePattern(graph);
}

void ReshapeCalcOfflinePass::RemoveReshapePattern(
    const std::unique_ptr<SSAGraph>& graph) {
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (node->AsStmt().op_type() != "reshape" &&
        node->AsStmt().op_type() != "reshape2")
      continue;
    auto outlinks = node->outlinks;
    bool has_extra_producers = false;
    for (auto& out_link : outlinks) {
      if (HasExtraProducers(
              graph.get(), out_link->arg()->name, {"reshape", "reshape2"})) {
        has_extra_producers = true;
        break;
      }
    }
    if (has_extra_producers) {
      LOG(WARNING)
          << "Unsupported for op output var containing multiple producers";
      continue;
    }

    std::set<const Node*> nodes2rm_;
    auto& reshape_instruct = node->AsStmt();
    auto* scope = reshape_instruct.op()->scope();
    auto op_desc = reshape_instruct.mutable_op_info();
    // Get reshape's input tensor
    auto input_var = scope->FindVar(op_desc->Input("X").front());
    auto input_t = input_var->GetMutable<lite::Tensor>();
    auto input_dims = input_t->dims();
    if (!input_t->persistable()) {
      LOG(WARNING) << "ReshapeCalcOfflinePass does not support input("
                   << op_desc->Input("X").front()
                   << ") that is not persistable";
      continue;
    }
    // Get reshape's shape
    if ((op_desc->HasInput("ShapeTensor") &&
         !op_desc->Input("ShapeTensor").empty()) ||
        (op_desc->HasInput("Shape") && !op_desc->Input("Shape").empty())) {
      LOG(WARNING) << "Unsupported Shape or ShapeTensor input for "
                      "reshape op.";
      continue;
    } else if (!op_desc->HasAttr("shape")) {
      LOG(WARNING) << "shape(attr) must be set for ReshapeCalcOfflinePass.";
      continue;
    }
    auto shape = op_desc->GetAttr<std::vector<int>>("shape");
    // Get reshape's output tensor
    auto out_var = scope->FindVar(op_desc->Output("Out").front());
    auto out_t = out_var->GetMutable<lite::Tensor>();
    auto output_shape = ValidateShape(shape, input_dims);
    out_t->CopyDataFrom(*input_t);
    out_t->Resize(DDim(output_shape));
    // Offline calc reshape, only retain output tensor as persistable
    // tensor
    out_t->set_persistable(true);
    auto reshape_outlinks = node->outlinks;
    for (auto& reshape_out_link : reshape_outlinks) {
      reshape_out_link->arg()->is_weight = true;
    }
    nodes2rm_.insert(node);
    GraphSafeRemoveNodes(graph.get(), nodes2rm_);
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(reshape_calc_offline_pass,
                  paddle::lite::mir::ReshapeCalcOfflinePass)
    .BindTargets({TARGET(kNNAdapter)});
