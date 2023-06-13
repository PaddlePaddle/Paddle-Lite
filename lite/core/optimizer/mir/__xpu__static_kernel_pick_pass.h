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
#ifdef LITE_WITH_XPU
#include "lite/backends/xpu/target_wrapper.h"
#endif
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
  void InitKernelPickInfo() {
    // clear kernel pick info to avoid crash when
    // init two models respectively
    xpu_input_type_.clear();
    xpu_output_type_.clear();
    xpu_special_op_.clear();
    xpu_use_int8_optimizer_ = false;
    xpu_use_fp16_optimizer_ = false;
    xpu_int8_compute_autotune_ = false;
    xpu_full_quantization_ = true;
    fetch_tensor_in_xpu_ = false;
  }

  const core::KernelPickFactor& kernel_pick_factors() const {
    return kernel_pick_factors_;
  }
  core::KernelPickFactor* mutable_kernel_pick_factors() {
    return &kernel_pick_factors_;
  }

 private:
  void Init() {
#ifdef LITE_WITH_XPU
    // get xpu device type
    int cur_dev_idx = 0;
    uint64_t cur_dev_attr = 0;
    XPU_CALL(xpu_current_device(&cur_dev_idx));
    XPU_CALL(xpu_device_get_attr(&cur_dev_attr, XPUATTR_MODEL, cur_dev_idx));
    if (cur_dev_attr <= 1) {
      VLOG(4) << "Currents XPU device : XPU1";
      xpu_disable_flag_ = "DISABLE_XPU1";
      xpu_device_version_ = "XPU1";
    } else if (cur_dev_attr >= 2 && cur_dev_attr <= 299) {
      VLOG(4) << "Currents XPU device : XPU2";
      xpu_disable_flag_ = "DISABLE_XPU2";
      xpu_device_version_ = "XPU2";
    } else if (cur_dev_attr >= 300 && cur_dev_attr <= 599) {
      VLOG(4) << "Currents XPU device : XPU3";
      xpu_disable_flag_ = "DISABLE_XPU3";
      xpu_device_version_ = "XPU3";
    } else {
      VLOG(4) << "invaid XPU device";
      xpu_disable_flag_ = "NONE";
    }
    // init quant type, encode precision
    CHECK(lite::TargetWrapperXPU::xpu_runtime_ptr)
        << "xpu_runtime_ptr null in pass";
    local_quant_ =
        GetBoolFromEnv("XPU_LOCAL_QUANT",
                       lite::TargetWrapperXPU::xpu_runtime_ptr->local_quant);
    encode_precision_ = GetStringFromEnv(
        "XPU_ENCODER_PRECISION",
        lite::TargetWrapperXPU::xpu_runtime_ptr->multi_encoder_precision);
    xpu_int8_compute_autotune_ = GetBoolFromEnv("XPU_INT8_AUTOTUNE", false);
    xpu_full_quantization_ = GetBoolFromEnv("XPU_FULL_QUANTIZATION", true);
    fetch_tensor_in_xpu_ = GetBoolFromEnv("FETCH_TENSOR_IN_XPU", false);
    kernel_use_host_ = false;
#endif
  }

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
        if (instruct.op_info()->Type() == "fetch" &&
            kernel.target() == TARGET(kXPU) && !fetch_tensor_in_xpu_) {
          score = 0;
          VLOG(4)
              << "By default, the output tensor of fetch op is not on the xpu";
        }
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

      // temp add : xpu kernel multiclass output in cpu.
      if (instruct.op_info()->Type() != "multiclass_nms3" && kernel_use_host_ &&
          kernel.target() == TARGET(kXPU)) {
        score = 0;
      }

      if (instruct.op_info()->Type() == "multiclass_nms3") {
        kernel_use_host_ = true;
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

    return final_score;
  }

  // Compatible for PrecisionType.
  // In the process of choosing kernel, fp16 and fp32 are
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
  void DataPrecisionDicide(const std::unique_ptr<SSAGraph>& graph);
  bool ForceUsePrecision(size_t* score,
                         const lite::KernelBase& kernel,
                         const paddle::lite::mir::Node::Stmt& instruct);
  void GetScore(PrecisionType precision, size_t* score_tmp);

  void NodeInputPrecision(lite::mir::Node* node,
                          const std::unique_ptr<SSAGraph>& graph);
  void InplaceNodeInputPrecision(lite::mir::Node* node);
  void SpecialNodeInputPrecision(lite::mir::Node* node,
                                 const bool collect_int8,
                                 const bool collect_fp16,
                                 bool* has_collected);

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
  void GeneralInt8OpScore(lite::mir::Node* node,
                          const lite::KernelBase& kernel,
                          bool* type_match,
                          size_t* score);
  void SetEnableInt8Attribute(const std::unique_ptr<SSAGraph>& graph);
  void strategiesInt8OP(lite::mir::Node* op_node, bool* quant_int8);
  void strategiesconcatOP(const std::unique_ptr<SSAGraph>& graph,
                          lite::mir::Node* op_node,
                          bool* quant_int8);
  void SliceForceNotUseXPU(lite::mir::Node* node,
                           const lite::KernelBase& kernel,
                           bool* type_match,
                           size_t* score);

 private:
  core::KernelPickFactor kernel_pick_factors_;

  bool xpu_use_fp16_optimizer_{false};
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
                                              "unsqueeze2",
                                              "flatten_contiguous_range"};
  bool xpu_use_int8_optimizer_{false};
  const std::set<std::string> xpu_int8_special_op_{"__xpu__fc",
                                                   "__xpu__conv2d"};
  // Temp add: owing to slim bug,this op is not support int8 compute.
  const std::set<std::string> xpu_disable_int_op_{"matmul_v2",
                                                  "conv2d_transpose"};

  const std::set<std::string> xpu_int8_general_op_not_need_sacale_{
      "nearest_interp",
      "nearest_interp_v2",
      "transpose",
      "transpose2",
      "split",
      "clip",
      "slice",
      "shape"};

  const std::set<std::string> xpu_int8_general_op_{"pool2d",
                                                   "elementwise_add",
                                                   "elementwise_mul",
                                                   "concat",
                                                   "reduce_mean",
                                                   "bilinear_interp",
                                                   "bilinear_interp_v2",
                                                   "nearest_interp",
                                                   "nearest_interp_v2",
                                                   "transpose",
                                                   "transpose2",
                                                   "split",
                                                   "clip",
                                                   "slice",
                                                   "shape"};

  bool local_quant_{false};
  std::string encode_precision_;
  bool kernel_use_host_ = false;
  bool xpu_int8_compute_autotune_{false};
  bool xpu_full_quantization_{true};
  bool fetch_tensor_in_xpu_{false};
  std::string xpu_device_version_{};
};

}  // namespace mir
}  // namespace lite
}  // namespace paddle
