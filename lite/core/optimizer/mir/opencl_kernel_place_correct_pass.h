// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <utility>
#include <vector>

#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * Correct the place of the variables in the SSAGrpah, it will inference the
 * variables' place by the kernels outputs them.
 */
class OpenCLKernelPlaceCorrectPass : public ProgramPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

 private:
  void CorrectArgumentPlace(SSAGraph* graph) {
    for (auto& x : graph->StmtTopologicalOrder()) {
      auto& inst = x->AsStmt();
      // deal with inputs
      VLOG(4) << "checking op " << inst.op_info()->Repr();

      auto in = x->inlinks.front();
      if (!in) {
        continue;
      }
      auto out = x->outlinks.front();
      const auto& op_type = inst.op_type();

      if (op_type == "softmax") {
        const auto op = inst.op();
        const auto* op_info = inst.op_info();
        auto var_names = op_info->output_names();
        CHECK_EQ(var_names.size(), 1);
        auto* scope = op->scope();
        auto* var = scope->FindVar(var_names[0]);
        if (var == nullptr) {
          LOG(WARNING) << "var is nullptr! var_name: " << var_names[0];
          return;
        }

        const auto& tensor = var->Get<Tensor>();
        const auto dims = tensor.dims();
        int axis = -1;
        if (op_info->HasAttr("axis")) {
          axis = op_info->GetAttr<int>("axis");
        }
        axis = axis >= 0 ? axis : axis + dims.size();
        VLOG(4) << "dims: " << dims << "\t dims[axis]: " << dims[axis];

        // OpenCL parallelism is low at this case,
        // so we use host backend kernel.
        const int thres = 500;
        if (dims[axis] > thres) {
          TargetType new_target = TARGET(kARM);
          const auto& valid_places = graph->valid_places();
          for (const auto& place : valid_places) {
            if (place.target == TARGET(kARM)) {
              new_target = place.target;
              break;
            } else if (place.target == TARGET(kX86)) {
              new_target = place.target;
              break;
            }
          }
          VLOG(4) << string_format(
              "Correct opencl %s kernel & tensor's place from %s to %s.",
              op_type.c_str(),
              TargetToStr(inst.place().target).c_str(),
              TargetToStr(new_target).c_str());
          UpdateTarget(inst, new_target);
          UpdateTensor(inst, in, out, new_target);
        }
      }
    }
  }

  // Update me's kUnk fields by other's fields.
  void UpdateTarget(mir::Node::Stmt& inst, TargetType new_target) {  // NOLINT
    auto new_place = inst.place();

    new_place.target = new_target;
    if (new_target == TargetType::kARM) {
      new_place.precision = PrecisionType::kFloat;
      new_place.layout = DataLayoutType::kNCHW;
    }

    if (new_target == TargetType::kX86) {
      new_place.precision = PrecisionType::kFloat;
      new_place.layout = DataLayoutType::kNCHW;
    }

    if (new_target == TargetType::kHost) {
      new_place.precision = PrecisionType::kFloat;
      new_place.layout = DataLayoutType::kNCHW;
    }

    std::vector<Place> places;
    places.push_back(new_place);
    inst.ResetKernels(places);
  }

  void UpdateTensor(mir::Node::Stmt& inst,  // NOLINT
                    Node* in,
                    Node* out,
                    TargetType new_target = TargetType::kUnk) {
    auto get_argname = [&](
        const std::string& node_name,
        const std::map<std::string, std::vector<std::string>>& argname_map)
        -> std::string {
          for (auto& ele : argname_map) {
            auto it =
                std::find(ele.second.begin(), ele.second.end(), node_name);
            if (it != ele.second.end()) return ele.first;
          }
          return "";
        };

    std::string arg_name =
        get_argname(out->AsArg().name, inst.op_info()->outputs());
    std::string in_name =
        get_argname(in->AsArg().name, inst.op_info()->inputs());

    auto type = inst.picked_kernel().GetInputDeclType(in_name);
    auto tmp_ptype = in->AsArg().type->precision();
    auto tmp_target = type->target();
    auto tmp_layout = type->layout();

    if (new_target == TargetType::kARM) {
      tmp_target = TargetType::kARM;
      tmp_ptype = PrecisionType::kFloat;
      tmp_layout = DataLayoutType::kNCHW;
    }

    if (new_target == TargetType::kX86) {
      tmp_target = TargetType::kX86;
      tmp_ptype = PrecisionType::kFloat;
      tmp_layout = DataLayoutType::kNCHW;
    }

    if (new_target == TargetType::kHost) {
      tmp_target = TargetType::kHost;
      tmp_ptype = PrecisionType::kFloat;
      tmp_layout = DataLayoutType::kNCHW;
    }

    out->AsArg().type =
        LiteType::GetTensorTy(tmp_target, tmp_ptype, tmp_layout);
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
