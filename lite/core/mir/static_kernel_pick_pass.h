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
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "lite/core/mir/pass.h"
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
  size_t KernelGrade(
      const lite::mir::Node::Stmt& instruct,
      const lite::KernelBase& kernel,
      const std::vector<Place>& places,
      const std::unordered_map<std::string, PrecisionType>& in_types,
      const std::unordered_map<std::string, PrecisionType>& out_types,
      const std::vector<std::string>& in_names,
      const std::vector<std::string>& out_names) {
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
      size_t score{};

      // The more important factor comes first
      if (kernel_pick_factors_.IsTargetConsidered() &&
          (place.target == kernel.target() || kernel.target() == TARGET(kAny) ||
           place.target == TARGET(kAny))) {
        score += kMax /
                 static_cast<int>(core::KernelPickFactor::Factor::TargetFirst);
      }
      VLOG(4) << "[score s1]:" << score;
      if (kernel_pick_factors_.IsPrecisionConsidered() &&
          (place.precision == kernel.precision() ||
           kernel.precision() == PRECISION(kAny) ||
           place.precision == PRECISION(kAny))) {
        // score skipped, if kernel is int8, but op is not int8
        if (!(kernel.precision() == PRECISION(kInt8) &&
              !instruct.op_info()->HasAttr("enable_int8"))) {
          score += kMax / static_cast<int>(
                              core::KernelPickFactor::Factor::PrecisionFirst);
        }
      }
      VLOG(4) << "[score s2]:" << score;
      if (kernel_pick_factors_.IsDataLayoutConsidered() &&
          (place.layout == kernel.layout() ||
           kernel.layout() == DATALAYOUT(kAny) ||
           place.layout == DATALAYOUT(kAny))) {
        score += kMax / static_cast<int>(
                            core::KernelPickFactor::Factor::DataLayoutFirst);
      }
      VLOG(4) << "[score s3]:" << score;

      // add new rules for precision: When the input types are consistent with
      // kernel's input types  and the output types are consistent with kernel's
      // output types. Select the kernel of the precision. Note that this
      // strategy is not compatible with quantization, so skip quantization op.
      if (!instruct.op_info()->HasAttr("enable_int8")) {
        bool type_match = true;
        for (size_t i = 0; i < in_names.size(); ++i) {
          std::string tmp;
          CHECK(instruct.op_info()->GetInputArgname(in_names[i], &tmp));
          if (in_types.count(in_names[i]) &&
              in_types.at(in_names[i]) !=
                  kernel.GetInputDeclType(tmp)->precision()) {
            type_match = false;
          }
        }
        for (size_t i = 0; i < out_names.size(); ++i) {
          std::string tmp;
          CHECK(instruct.op_info()->GetOutputArgname(out_names[i], &tmp));
          if (out_types.count(out_names[i]) &&
              out_types.at(out_names[i]) !=
                  kernel.GetOutputDeclType(tmp)->precision()) {
            type_match = false;
          }
        }
        if (type_match) {
          score *= 2;
        }
        VLOG(4) << "[score s4]:" << score;
      }

      if (weight * score > final_score) {
        final_score = weight * score;
        winner_place = place;
      }
    }

    VLOG(4) << "[score(final)]:" << final_score;
    VLOG(2) << "-------- pick summary for " << instruct.op_type()
            << " --------";
    VLOG(2) << " ===> winner_place():" << PrecisionToStr(winner_place.precision)
            << " " << DataLayoutToStr(winner_place.layout) << " "
            << TargetToStr(winner_place.target);
    VLOG(2) << " ===> kernel.place():"
            << PrecisionToStr(kernel.place().precision) << " "
            << DataLayoutToStr(kernel.place().layout) << " "
            << TargetToStr(kernel.place().target);
    VLOG(4) << "kernel.op_type():" << kernel.op_type();
    VLOG(4) << "kernel picker factors:" << kernel_pick_factors_;
    VLOG(4) << "kernel place:" << kernel.place().DebugString();
    VLOG(4) << "winner_picker place:" << winner_place.DebugString();
    VLOG(4) << "------------------------------";

    // The data layout is not considered, for the input and output arguments
    // might have different data layout.
    // TODO(Superjomn) reconsider the idea of taking the data layout as a kernel
    // specification.
    return final_score;
  }

 private:
  core::KernelPickFactor kernel_pick_factors_;
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
