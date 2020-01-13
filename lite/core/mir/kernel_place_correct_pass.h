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
 * Correct the place of the variables in the SSAGrpah, it will inference the
 * variables' place by the kernels outputs them.
 */
class KernelPlaceCorrectPass : public DebugPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

 private:
  void CorrectArgumentPlace(SSAGraph* graph) {
    auto& valid_places = graph->valid_places();
    auto valid_places_has_target = [&](TargetType t) -> bool {
      for (auto& p : valid_places) {
        if (p.target == t) {
          return true;
        }
      }
      return false;
    };
    std::map<std::string, bool> lite_with_targets{
        {"kOpenCL", valid_places_has_target(TARGET(kOpenCL))},
        {"kFPGA", valid_places_has_target(TARGET(kFPGA))}};
    VLOG(4) << "lite_with_targets['kOpenCL']:" << lite_with_targets["kOpenCL"];
    VLOG(4) << "lite_with_targets['kFPGA']:" << lite_with_targets["kFPGA"];

    VLOG(3) << "param-type-registry:\n" << ParamTypeRegistry::Global();
    for (auto& x : graph->StmtTopologicalOrder()) {
      auto& inst = x->AsStmt();
      // The IoCopyOp is a tool operator, it won't support the type inference.
      // in fpga, we has io_copy+cali+layout tool ops, so we need type inference
      // for
      // tool operator
      if ((!lite_with_targets["kFPGA"]) && (!lite_with_targets["kOpenCL"])) {
        VLOG(3) << "inst.op_type() == 'io_copy', continue";
        if (inst.op_type() == "io_copy") continue;
      }
      // deal with inputs
      VLOG(4) << "checking op " << inst.op_info()->Repr();

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

      bool need_correct_place = true;

      std::vector<TargetType> in_types;
      std::vector<TargetType> out_types;
      for (auto* x_in : x->inlinks) {
        std::string node_name = x_in->AsArg().name;
        std::string arg_name = get_argname(node_name, inst.op_info()->inputs());
        CHECK(arg_name.size() > 0) << "can not found op arguments for node "
                                   << node_name;
        VLOG(4) << "-- input arg_name:" << arg_name << " "
                << "-- node name:" << node_name;
        auto type = inst.picked_kernel().GetInputDeclType(arg_name);
        if (!x_in->AsArg().type) {
          need_correct_place &= false;
        } else {
          if (in_types.empty()) {
            in_types.push_back(x_in->AsArg().type->target());
          } else {
            if (in_types[0] != x_in->AsArg().type->target()) {
              need_correct_place &= false;
            }
          }
        }
      }

      for (auto* x_out : x->outlinks) {
        std::string node_name = x_out->AsArg().name;
        std::string arg_name =
            get_argname(node_name, inst.op_info()->outputs());
        CHECK(arg_name.size() > 0) << "can not found op arguments for node "
                                   << node_name << " in Inst "
                                   << inst.op_type();
        VLOG(4) << "-- output arg_name " << arg_name;
        auto type = inst.picked_kernel().GetOutputDeclType(arg_name);
        if (!x_out->AsArg().type) {
          need_correct_place &= false;
        } else {
          if (out_types.empty()) {
            out_types.push_back(x_out->AsArg().type->target());
          } else {
            if (out_types[0] != x_out->AsArg().type->target()) {
              need_correct_place &= false;
            }
          }
        }
      }

      auto this_type = inst.picked_kernel().target();
      bool io_target_same = (in_types[0] == out_types[0]);
      need_correct_place &= (io_target_same && (in_types[0] != this_type));
      if (need_correct_place) {
        // update this kernel's valid place;
        UpdateTarget(inst, in_types[0]);
      }
    }
  }

  // Update me's kUnk fields by other's fields.
  void UpdateTarget(mir::Node::Stmt& inst, TargetType new_target) {  // NOLINT
    auto new_place = inst.place();
    new_place.target = new_target;
    std::vector<Place> places;
    places.push_back(new_place);
    inst.ResetKernels(places);
  }
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
