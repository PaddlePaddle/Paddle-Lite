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

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * StaticKernelPickPass is a simple strategy for picking the kernel for each
 * Operator using operator developer defined rule, there are many other tactics
 * such as considering IO or kernel execution latency and we will implement them
 * latter.
 *
 * There are two argument for this pass:
 * - place, the target place.
 * - kernel_pick_factors, the factors to consider in picking kernels.
 * Set them first before execute the pass.
 */
class StaticKernelPickPass : public mir::StmtPass {
 public:
  void Apply(const std::unique_ptr<SSAGraph>& graph) override;

  const core::KernelPickFactor& kernel_pick_factors() const {
    return kernel_pick_factors_;
  }
  core::KernelPickFactor* mutable_kernel_pick_factors() {
    return &kernel_pick_factors_;
  }

 private:
  // Score the kernel.
  size_t KernelGrade(lite::mir::Node* node,
                     const lite::KernelBase& kernel,
                     const std::vector<Place>& places,
                     const std::map<std::string, PrecisionType>& in_types,
                     const std::map<std::string, PrecisionType>& out_types,
                     const std::vector<std::string>& in_names,
                     const std::vector<std::string>& out_names) {
    const auto& instruct = node->AsStmt();
    CHECK_GT(places.size(), static_cast<size_t>(0)) << "valid_places is empty.";
    float final_score{-1.};
    Place winner_place{places[0]};
    const int kMax =
        (std::numeric_limits<core::KernelPickFactor::value_type>::max)();
    size_t place_size = places.size();

    // NOTE: We compare kernel's place with place in valid_places to select the
    // best match place
    //       The place's order in valid_places array decide the user's
    //       preference
    // final_score = weight * socre
    // weight: The weight is compute with (valid_places.size() - i) /
    // valid_places.size() as default.
    //         where i is the place's index in valid_places array.
    // score:  score is the weighted sum of target、percision and layout
    for (size_t i = 0; i < place_size; ++i) {
      const auto& place = places[i];
      float weight = static_cast<float>(place_size - i) / place_size;
      VLOG(4) << "current place is " << place.DebugString() << ", idx : " << i
              << ", weight : " << weight;
      size_t score{};

      // The more important factor comes first
      if (kernel_pick_factors_.IsTargetConsidered() &&
          (place.target == kernel.target() || kernel.target() == TARGET(kAny) ||
           place.target == TARGET(kAny))) {
        size_t target_score =
            kMax /
            static_cast<int>(core::KernelPickFactor::Factor::TargetFirst);
        score += target_score;
        VLOG(4) << "[TargetConsidered score]:" << target_score;
      }
      VLOG(4) << "[score s1]:" << score;
      if (kernel_pick_factors_.IsPrecisionConsidered() &&
          (place.precision == kernel.precision() ||
           kernel.precision() == PRECISION(kAny) ||
           place.precision == PRECISION(kAny))) {
        // score skipped, if kernel is int8, but op is not int8
        if (!(kernel.precision() == PRECISION(kInt8) &&
              !instruct.op_info()->HasAttr("enable_int8"))) {
          size_t precision_score =
              kMax /
              static_cast<int>(core::KernelPickFactor::Factor::PrecisionFirst);
          score += precision_score;
          VLOG(4) << "[PrecisionConsidered score]:" << precision_score;
        }
      }
      VLOG(4) << "[score s2]:" << score;
      if (kernel_pick_factors_.IsDataLayoutConsidered() &&
          (place.layout == kernel.layout() ||
           kernel.layout() == DATALAYOUT(kAny) ||
           place.layout == DATALAYOUT(kAny))) {
        size_t datalayout_score =
            kMax /
            static_cast<int>(core::KernelPickFactor::Factor::DataLayoutFirst);
        score += datalayout_score;
        VLOG(4) << "[DataLayoutConsidered score]:" << datalayout_score;
      }
      VLOG(4) << "[score s3]:" << score;

      // add new rules for precision: When the input types are consistent with
      // kernel's input types, select the kernel of the precision. However, if
      // the op is feed, we should compare the output precision type.
      // Note that this strategy is not compatible with quantization, so skip
      // quantization op.
      if (!instruct.op_info()->HasAttr("enable_int8")) {
        bool type_match = true;
        if (instruct.op_type() == "feed") {
          for (size_t i = 0; i < out_names.size(); ++i) {
            std::string tmp;
            CHECK(instruct.op_info()->GetOutputArgname(out_names[i], &tmp));
            if (out_types.count(out_names[i]) &&
                out_types.at(out_names[i]) !=
                    kernel.GetOutputDeclType(tmp)->precision()) {
              type_match = false;
            }
          }
        } else {
          for (size_t i = 0; i < in_names.size(); ++i) {
            std::string tmp;
            CHECK(instruct.op_info()->GetInputArgname(in_names[i], &tmp));
            if (in_types.count(in_names[i])) {
              if (!PrecTypeCompatible(
                      in_types.at(in_names[i]),
                      kernel.GetInputDeclType(tmp)->precision())) {
                type_match = false;
              } else {
                score += 1;
              }
            }
          }
        }
        if (type_match) {
          score *= 2;
          VLOG(4) << "[Input precision compatible]: *2";
        }
        VLOG(4) << "[score s4]:" << score;
      }

      // add new rules for datatype: When the input types are consistent with
      // kernel's input types, select the kernel of the datatype.
      if (instruct.op_info()->Type() != "conditional_block" &&
          instruct.op_info()->Type() != "while" &&
          instruct.op_info()->Type() != "subgraph") {
        bool datatype_match = true;
        for (auto* in : node->inlinks) {
          if (!in->IsArg()) continue;
          if (in->AsArg().name == "feed" || in->AsArg().is_persist) continue;
          std::string argname;
          instruct.op_info()->GetInputArgname(in->AsArg().name, &argname);
          VLOG(5) << "intput var name : " << in->AsArg().name;
          // only when datatype is LOD_TENSOR, LOD_TENSOR_ARRAY, STEP_SCOPES,
          // the type pointer is not null;
          if (in->AsArg().type) {
            VLOG(5) << "input datatype : "
                    << static_cast<int>(in->AsArg().type->id());
            VLOG(5) << "kernel bind datatype : "
                    << static_cast<int>(kernel.GetInputDeclType(argname)->id());
            if (static_cast<int>(in->AsArg().type->id()) !=
                static_cast<int>(kernel.GetInputDeclType(argname)->id()))
              datatype_match = false;
          } else {
            datatype_match = false;
          }
        }
        if (datatype_match) {
          score *= 2;
          VLOG(4) << "[Input datatype compatible]: *2";
        }
        VLOG(4) << "[score s5]:" << score;
      }

      if (kernel.place().target == TARGET(kOpenCL)) {
        if (instruct.op_type() == "matmul" ||
            instruct.op_type() == "matmul_v2") {
          bool input_target_match = false;
          int persistable_weights = 0;
          int input_match_num = 0;
          for (auto* in : node->inlinks) {
            if (!in->IsArg()) continue;
            if (in->AsArg().name == "feed") continue;
            VLOG(4) << "persistable attr is: " << in->AsArg().is_persist;
            VLOG(4) << "is_weight attr is: " << in->AsArg().is_weight;
            std::string argname;
            instruct.op_info()->GetInputArgname(in->AsArg().name, &argname);
            VLOG(4) << "input var name : " << in->AsArg().name;
            if (in->AsArg().is_weight || in->AsArg().is_persist)
              persistable_weights++;
            if (persistable_weights > 0 &&
                kernel.GetInputDeclType(argname)->target() == TARGET(kHost)) {
              input_target_match = true;
            } else if (kernel.GetInputDeclType(argname)->target() ==
                       TARGET(kOpenCL)) {
              input_match_num++;
            }
          }
          if (persistable_weights == 0 && input_match_num == 2) {
            input_target_match = true;
          }
          if (input_target_match) {
            score *= 2;
            VLOG(4) << "[Input target compatible]: *2";
          }
          VLOG(4) << "[score s6]:" << score;
        }
      }

      if (weight * score > final_score) {
        final_score = weight * score;
        winner_place = place;
      }
    }

    if (kernel.target() == TARGET(kFPGA)) {
      VLOG(4) << "alias:" << kernel.alias();
      /**
       * we want to use fpga kernel as much as possible, so we give it a very
       *high score,
       * so this kernel can be picked, it may be not the best option, and we
       *shall correct
       * it in kernel_place_correct_pass
       *
       * 4000 is a arbitrary high score that can purpress all the other kernels.
       **/
      final_score = 4000;
      for (size_t i = 0; i < in_names.size(); ++i) {
        std::string tmp;
        CHECK(instruct.op_info()->GetInputArgname(in_names[i], &tmp));
        if (in_types.count(in_names[i]) &&
            in_types.at(in_names[i]) ==
                kernel.GetInputDeclType(tmp)->precision()) {
          final_score += 100;  // multiple inputs pick the most matched one;
        }
      }

      for (size_t i = 0; i < out_names.size(); ++i) {
        std::string tmp;
        CHECK(instruct.op_info()->GetOutputArgname(out_names[i], &tmp));

        VLOG(4) << tmp << " == "
                << PrecisionToStr(kernel.GetOutputDeclType(tmp)->precision());
        if (out_types.count(out_names[i]) > 0) {
          VLOG(4) << "decType: "
                  << PrecisionToStr(kernel.GetOutputDeclType(tmp)->precision());
          VLOG(4) << "cout:" << out_types.count(out_names[i]) << " type_name: "
                  << PrecisionToStr(out_types.at(out_names[i]));
        }

        if (out_types.count(out_names[i]) &&
            out_types.at(out_names[i]) ==
                kernel.GetOutputDeclType(tmp)->precision()) {
          final_score += 100;  // multiple outputs pick the most matched one;
        }
      }
    }

    VLOG(2) << "-------- score summary for candidate kernel : "
            << kernel.summary() << " --------";
    VLOG(2) << " ===> winner_place():" << PrecisionToStr(winner_place.precision)
            << " " << DataLayoutToStr(winner_place.layout) << " "
            << TargetToStr(winner_place.target);
    VLOG(2) << " ===> kernel.place():"
            << PrecisionToStr(kernel.place().precision) << " "
            << DataLayoutToStr(kernel.place().layout) << " "
            << TargetToStr(kernel.place().target);
    VLOG(2) << "kernel.op_type():" << kernel.op_type();
    VLOG(2) << "kernel picker factors:" << kernel_pick_factors_;
    VLOG(2) << "winner_picker place:" << winner_place.DebugString();
    VLOG(2) << "[score(final)]:" << final_score;
    VLOG(2) << "------------------------------";

    // The data layout is not considered, for the input and output arguments
    // might have different data layout.
    // TODO(Superjomn) reconsider the idea of taking the data layout as a kernel
    // specification.
    return final_score;
  }

  // Compatible for PrecisionType.
  // For cuda, in the process of choosing kernel, fp16 and fp32 are compatiable.
  // If kernel's declared type is kAny, it is matched.
  bool PrecTypeCompatible(const PrecisionType& p1, const PrecisionType& p2) {
    if (p1 == p2 || p2 == PRECISION(kAny)) {
      return true;
    } else if ((p1 == PRECISION(kFP16) || p1 == PRECISION(kFloat)) &&
               (p2 == PRECISION(kFP16) || p2 == PRECISION(kFloat))) {
      return true;
    } else {
      return false;
    }
  }

 private:
  core::KernelPickFactor kernel_pick_factors_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
