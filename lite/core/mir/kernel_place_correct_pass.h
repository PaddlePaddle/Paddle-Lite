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

      auto in = x->inlinks.front();
      if (!in) {
        break;
      }
      auto out = x->outlinks.front();
      auto p = in->AsArg().type->precision();

      std::string node_name = out->AsArg().name;
      std::string arg_name = get_argname(node_name, inst.op_info()->outputs());

      auto op_type = inst.op_type();

      if (op_type == "reshape" || op_type == "reshape2") {
        for (auto* x_in : x->inlinks) {
          std::string in_name =
              get_argname(x_in->AsArg().name, inst.op_info()->inputs());
          if (in_name == "X") {
            in = x_in;
          }
        }

        p = in->AsArg().type->precision();
        if (p != PrecisionType::kFP16) {
          UpdateTarget(inst, TargetType::kHost);
          UpdateTensor(inst, in, out, TargetType::kHost);
        }
      }

      if (inst.op_type() == "fetch") {
        UpdateTarget(inst, TargetType::kFPGA);
      }

      if (inst.op_type() == "split" || inst.op_type() == "transpose" ||
          inst.op_type() == "transpose2") {
        if (p != PrecisionType::kFP16) {
          UpdateTarget(inst, TargetType::kARM);
          for (auto* x_out : x->outlinks) {
            UpdateTensor(inst, in, x_out, TargetType::kARM);
          }
        }
      }

      if (inst.op_type() == "concat") {
        // std::cout << "concat target:" <<
        // TargetRepr(inst.kernels()[0]->target())
        //           << std::endl;
        // std::cout << "concat p:"
        //           << PrecisionToStr(inst.kernels()[0]->precision())
        //           << std::endl;
        if (p != PrecisionType::kFP16) {
          UpdateTarget(inst, TargetType::kARM);
          UpdateTensor(inst, in, out, TargetType::kARM);
        }
      }

      // if (inst.op_type() == "elementwise_mul") {

      //   for (auto* x_in : x->inlinks) {

      //     std::string in_name = get_argname(x_in->AsArg().name,
      //     inst.op_info()->inputs());
      //     std::cout << "name: " << x_in->AsArg().name  << std::endl;
      //     std::cout << "in_name: " << in_name  << std::endl;
      //     if (in_name == "Y") {
      //       in = x_in;
      //       std::cout << "found y \n";
      //       // exit(-1);
      //     }
      //   }

      //   if ( p != PrecisionType::kFP16) {
      //     UpdateTarget(inst, TargetType::kARM);
      //     UpdateTensor(inst, in, out, TargetType::kARM);
      //   }
      // }

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

        // std::cout << arg_name <<" is weight:: " <<
        // std::to_string(x_in->AsArg().is_weight)
        //     << "     is persist: " <<
        //     std::to_string(x_in->AsArg().is_persist) << std::endl;

        // std::cout << " type: "<< inst.op_type() << std::endl;

        if (!x_in->AsArg().is_weight) {
          auto p = x_in->AsArg().type->precision();
          auto t = x_in->AsArg().type->target();
          auto l = x_in->AsArg().type->layout();
          // std::cout << "p:" << PrecisionToStr(p) << std::endl;
          // std::cout << "t:" << TargetRepr(t) << std::endl;
          // std::cout << "layout:" << DataLayoutToStr(l) << std::endl;
        }

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
        // UpdateTarget(inst, in_types[0]);
      }
    }
  }

  // Update me's kUnk fields by other's fields.
  void UpdateTarget(mir::Node::Stmt& inst, TargetType new_target) {  // NOLINT
    // std::cout << "1 kernels: " << std::to_string(inst.kernels().size()) <<
    // std::endl;
    auto new_place = inst.place();

    new_place.target = new_target;
    if (new_target == TargetType::kARM) {
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
    // std::cout << "2 kernels: " << std::to_string(inst.kernels().size()) <<
    // std::endl;
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
