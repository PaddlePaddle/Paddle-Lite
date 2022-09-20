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
#include <set>
#include <string>
#include <vector>
#include "lite/core/optimizer/mir/pass.h"
#include "lite/core/types.h"

namespace paddle {
namespace lite {
namespace mir {

/*
 * XPUStaticKernelPickPass is a simple strategy for picking the kernel for each
 * Operator using operator developer defined rule, there are many other tactics
 * such as considering IO or kernel execution latency and we will implement them
 * latter.
 *
 * There are two argument for this pass:
 * - place, the target place.
 * - kernel_pick_factors, the factors to consider in picking kernels.
 * Set them first before execute the pass.
 */
class XPUStaticKernelPickPass : public mir::StmtPass {
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

#ifdef LITE_WITH_XPU
      bool type_match = false;
      GradeXPUKernelScore(node,
                          kernel,
                          instruct,
                          in_names,
                          out_names,
                          in_types,
                          out_types,
                          &score,
                          &type_match);
      if (type_match) {
        score *= 2;
        VLOG(4) << "[Input/Output precision compatible]: *2";
      }
      VLOG(4) << "[score s4]:" << score;
#endif

      // add new rules for datatype: When the input types are consistent
      // with
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
          // only when datatype is LOD_TENSOR, LOD_TENSOR_ARRAY,
          // STEP_SCOPES,
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

      if (weight * score > final_score) {
        final_score = weight * score;
        winner_place = place;
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
    VLOG(4) << "kernel.op_type():" << kernel.op_type();
    VLOG(4) << "kernel picker factors:" << kernel_pick_factors_;
    VLOG(4) << "winner_picker place:" << winner_place.DebugString();
    VLOG(4) << "[score(final)]:" << final_score;
    VLOG(4) << "------------------------------";

    // The data layout is not considered, for the input and output arguments
    // might have different data layout.
    // TODO(Superjomn) reconsider the idea of taking the data layout as a
    // kernel
    // specification.
    return final_score;
  }

  // Compatible for PrecisionType.
  // For cuda, in the process of choosing kernel, fp16 and fp32 are
  // compatiable.
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
#ifdef LITE_WITH_XPU
  void DataPrecisionDicide(const std::unique_ptr<SSAGraph>& graph);
  bool ForceUsePrecision(size_t* score,
                         const lite::KernelBase& kernel,
                         const paddle::lite::mir::Node::Stmt& instruct);
  void GetScore(PrecisionType precision, size_t* score_tmp);

  void NodeInputPrecision(lite::mir::Node* node,
                          const std::unique_ptr<SSAGraph>& graph);
  void InplaceNodeInputPrecision(lite::mir::Node* node);
  void SpecialNodeInputPrecision(lite::mir::Node* node);

  void NodeOutputPrecision(const std::unique_ptr<SSAGraph>& graph,
                           lite::mir::Node* node);
  void InplaceNodeOutputPrecision(lite::mir::Node* node);
  void SpecialNodeOutputPrecision(
      const std::unique_ptr<SSAGraph>& graph,
      lite::mir::Node* node,
      const std::unique_ptr<lite::KernelBase>& kernel);

  void SpecialOpScore(lite::mir::Node* node,
                      const lite::KernelBase& kernel,
                      bool* type_match,
                      size_t* score);
  void GetXPUDeviceType();
  void InplaceOpScore(lite::mir::Node* node,
                      const lite::KernelBase& kernel,
                      bool* type_match,
                      size_t* score);
  void GradeXPUKernelScore(
      lite::mir::Node* node,
      const lite::KernelBase& kernel,
      const paddle::lite::mir::Node::Stmt& instruct,
      const std::vector<std::string>& in_names,
      const std::vector<std::string>& out_names,
      const std::map<std::string, PrecisionType>& in_types,
      const std::map<std::string, PrecisionType>& out_types,
      size_t* score,
      bool* type_match);
  void CollectXPUSpecialOPType(const std::unique_ptr<SSAGraph>& graph);
#endif

 private:
  core::KernelPickFactor kernel_pick_factors_;

  bool xpu_use_fp16_optimizer_{false};
#ifdef LITE_WITH_XPU
  std::multimap<std::string, std::vector<std::map<std::string, PrecisionType>>>
      xpu_input_type_{};
  std::map<std::string, PrecisionType> xpu_output_type_{};
  std::string xpu_disable_flag_{};
  const std::set<std::string> consider_cpu_op_{"cast"};
  std::set<std::string> xpu_special_op_{};
  const std::set<std::string> xpu_inplace_op_{"reshape",
                                              "reshape2",
                                              "flatten",
                                              "flatten2",
                                              "squeeze",
                                              "squeeze2",
                                              "unsqueeze",
                                              "unsqueeze2"};
  // int8
  bool xpu_use_int8_optimizer_{false};
  std::set<std::string> xpu_int8_special_op_{"__xpu__fc", "__xpu__conv2d"};
#endif
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
