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

#include "lite/core/optimizer/mir/__xpu__static_kernel_pick_pass.h"
#include <algorithm>
#include <cmath>
#include <list>
#include <map>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>
#include "lite/core/optimizer/mir/graph_visualize_pass.h"
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

bool XPUKernelScoreCmp(const std::pair<float, std::unique_ptr<KernelBase>>& a,
                       const std::pair<float, std::unique_ptr<KernelBase>>& b) {
  return a.first > b.first;
}

void XPUStaticKernelPickPass::Apply(const std::unique_ptr<SSAGraph>& graph) {
  Init();

  kernel_pick_factors_.ConsiderTarget();
  kernel_pick_factors_.ConsiderPrecision();
  kernel_pick_factors_.ConsiderDataLayout();
  CHECK(kernel_pick_factors_.any_factor_considered())
      << "kernel_pick_factors should be specified first";
  CHECK(graph) << "graph not valid";

  // Collect input data precision for each node in the graph
  // Collect XPU op type,which used in fp16/in8;
  DataPrecisionDicide(graph);
  if (xpu_use_int8_optimizer_ && xpu_full_quantization_ &&
      xpu_device_version_ != "XPU1") {
    SetEnableInt8Attribute(graph);
  }

  if (xpu_use_fp16_optimizer_ || xpu_use_int8_optimizer_) {
    CollectXPUSpecialOPType(graph);
    for (auto& node : graph->StmtTopologicalOrder()) {
      bool has_collected = false;
      if (!node->IsStmt()) continue;

      if (xpu_use_fp16_optimizer_ &&
          xpu_special_op_.count(node->AsStmt().op_type())) {
        SpecialNodeInputPrecision(
            node, false, xpu_use_fp16_optimizer_, &has_collected);
      }

      if (xpu_use_int8_optimizer_ &&
          (xpu_int8_special_op_.count(node->AsStmt().op_type()) ||
           xpu_int8_general_op_.count(node->AsStmt().op_type()))) {
        SpecialNodeInputPrecision(
            node, xpu_use_int8_optimizer_, false, &has_collected);
        if (has_collected) {
          continue;
        }
      }

      if (xpu_inplace_op_.count(node->AsStmt().op_type())) {
        continue;
      }

      NodeInputPrecision(node, graph);
    }
  }

  for (auto& node : graph->StmtTopologicalOrder()) {
    if (!node->IsStmt()) continue;
    if (xpu_inplace_op_.count(node->AsStmt().op_type()) == 0) {
      continue;
    }

    InplaceNodeInputPrecision(node);
  }

  // sort kernels by the factors.
  VLOG(2) << "graph block_idx: " << graph->blockIdx();
  VLOG(2) << "graph->mutable_nodes().size(): " << graph->mutable_nodes().size();
  size_t idx = 0;
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (!node->IsStmt()) continue;
    auto& instruct = node->AsStmt();
    VLOG(2) << "pick kernel for op : " << instruct.op_type() << ", in block "
            << graph->blockIdx() << ", idx : " << idx++;

    std::map<std::string, PrecisionType> in_types;
    std::map<std::string, PrecisionType> out_types;
    // these precision info store in __model__ file, if selected fp16 kernel,
    // the output precision should be changed.
    for (std::list<Node*>::iterator i = node->inlinks.begin();
         i != node->inlinks.end();
         ++i) {
      if ((*i)->arg()->type)
        in_types[(*i)->arg()->name] = (*i)->arg()->type->precision();
    }
    for (std::list<Node*>::iterator i = node->outlinks.begin();
         i != node->outlinks.end();
         ++i) {
      if ((*i)->arg()->type)
        out_types[(*i)->arg()->name] = (*i)->arg()->type->precision();
    }
    // Get candidate kernels
    std::vector<std::pair<float, std::unique_ptr<KernelBase>>> scored;
    CHECK(!instruct.kernels().empty()) << "No kernels found for "
                                       << instruct.op_type();

    VLOG(2) << "candidate kernels size:" << instruct.kernels().size();

    for (auto&& kernel : instruct.kernels()) {
      VLOG(2) << "current candidate kernel is: " << kernel->summary();
      VLOG(2) << "valid_places size is: " << graph->valid_places().size();
      if (!xpu_inplace_op_.count(instruct.op_type())) {
        if (!xpu_disable_int_op_.count(instruct.op_type())) {
          if (instruct.op_info()->HasAttr("enable_int8") &&
              instruct.op_info()->GetAttr<bool>("enable_int8") &&
              kernel->precision() != PrecisionType::kInt8 &&
              instruct.op_type() != "__xpu__multi_encoder") {
            VLOG(6) << "Ignore current kernel: " << kernel->summary()
                    << ", because we only want to pick int8 precision kernel.";
            continue;
          }
        }
      }

      float score = KernelGrade(node,
                                *kernel,
                                graph->valid_places(),
                                in_types,
                                out_types,
                                instruct.op_info()->input_names(),
                                instruct.op_info()->output_names());

      scored.emplace_back(score, std::move(kernel));
    }
    std::stable_sort(scored.begin(), scored.end(), XPUKernelScoreCmp);
    instruct.kernels().clear();

    if (xpu_use_fp16_optimizer_ || xpu_use_int8_optimizer_) {
      if ((xpu_use_fp16_optimizer_ &&
           xpu_special_op_.count(node->AsStmt().op_type())) ||
          (xpu_use_int8_optimizer_ &&
           (xpu_int8_special_op_.count(node->AsStmt().op_type()) ||
            xpu_int8_general_op_.count(instruct.op_type())))) {
        SpecialNodeOutputPrecision(graph, node, scored.front().second);
      } else if (xpu_inplace_op_.count(node->AsStmt().op_type())) {
        InplaceNodeOutputPrecision(node);
      } else {
        NodeOutputPrecision(graph, node);
      }
    }

    instruct.kernels().emplace_back(std::move(scored.front().second));
    VLOG(2) << "the final pick kernel is "
            << instruct.kernels().front()->summary() << "\n\n";
    instruct.mutable_op_info()->SetAttr<std::string>(
        "kernel_summary", instruct.kernels().front()->summary());
  }
}

void XPUStaticKernelPickPass::DataPrecisionDicide(
    const std::unique_ptr<SSAGraph>& graph) {
  if (GetStringFromEnv("XPUForceUseFP16", "false") == "true") {
    xpu_use_fp16_optimizer_ = false;
    VLOG(2) << "XPU force use data precision: FP16 ";
    return;
  }

  for (auto& place : graph->valid_places()) {
    if (place.precision == PrecisionType::kInt8 &&
        place.target == TargetType::kXPU) {
      xpu_use_int8_optimizer_ = true;
      VLOG(2) << "XPU auto use data precision: FP32/INT8.";
    }

    if (place.precision == PrecisionType::kFP16 &&
        place.target == TargetType::kXPU) {
      xpu_use_fp16_optimizer_ = true;
      VLOG(2) << "XPU auto use data precision: FP16/FP32.";
    }
  }
}

bool XPUStaticKernelPickPass::ForceUsePrecision(
    size_t* score,
    const lite::KernelBase& kernel,
    const paddle::lite::mir::Node::Stmt& instruct) {
  if (kernel.place().target != TARGET(kXPU)) {
    return false;
  }

  auto op_info = instruct.op_info();
  bool int8_quant =
      op_info->HasAttr("enable_int8") && op_info->GetAttr<bool>("enable_int8");
  bool int16_quant = op_info->HasAttr("enable_int16") &&
                     op_info->GetAttr<bool>("enable_int16");
  CHECK(!(int8_quant && int16_quant))
      << "You can only specify one quant type for an OP!";

  if (instruct.op_type() == "__xpu__fc") {
    if (int8_quant && kernel.alias() == "XPU_Int8_FP32_FP32" &&
        !encode_precision_.empty()) {
      *score *= 4;
      VLOG(6) << "__xpu__fc: force use PRECISON INT8: *4";
      return true;
    } else if (int16_quant && kernel.alias() == "XPUFC_INT16_FP32_FP32" &&
               !encode_precision_.empty()) {
      *score *= 4;
      VLOG(6) << "__xpu__fc: force use PRECISON INT16: *4";
      return true;
    } else if (local_quant_ && kernel.alias() == "XPU_FP32_LOCAL_QUANT") {
      *score *= 4;
      VLOG(6) << "__xpu__fc: force use LOCAL QUANT: *4";
      return true;
    } else if (encode_precision_ == "int31" &&
               kernel.alias() == "XPU_Real_kFloat") {
      *score *= 4;
      VLOG(6) << "__xpu__fc: force use PRECISON INT31: *4";
      return true;
    }
  }

  if (GetStringFromEnv("XPU_COMPUTE_PRECISION", "int16") == "int8") {
    if (kernel.alias() == "XPU_Int8_Int8_Int8") {
      *score *= 4;
      VLOG(6) << instruct.op_type() << ": force use PRECISON INT8: *4";
      return true;
    }
  }

  if (GetStringFromEnv("XPU_COMPUTE_PRECISION", "int16") == "int31") {
    if (kernel.alias() == "XPU_Real_kFloat") {
      *score *= 4;
      VLOG(6) << instruct.op_type() << ": force use PRECISON INT31: *4";
      return true;
    }
  }

  if (kernel.alias() == "XPU_Real_kFloat") {
    *score = 0;
    VLOG(6) << "By default,XPU not use PRECISION INT31, so not pick "
               "current kernel: "
            << kernel.summary();
    return true;
  }

  if (kernel.precision() == PRECISION(kInt8) &&
      !(op_info->HasAttr("enable_int8") &&
        op_info->GetAttr<bool>("enable_int8"))) {
    *score = 0;
    VLOG(6) << instruct.op_type() << "not pick int8 kernel,thanks to this op "
                                     "has not has attr:enable_int8,or "
                                     " attr:enable_int8 is false. "
            << kernel.summary();
    return true;
  }

  return false;
}

void XPUStaticKernelPickPass::GetScore(PrecisionType precision,
                                       size_t* score_tmp) {
  if (precision == PrecisionType::kInt8) {
    *score_tmp = *score_tmp > 11 ? *score_tmp : 11;
  } else if (precision == PrecisionType::kInt16) {
    *score_tmp = *score_tmp > 9 ? *score_tmp : 9;
  } else if (precision == PrecisionType::kFP16) {
    *score_tmp = *score_tmp > 7 ? *score_tmp : 7;
  } else if (precision == PrecisionType::kAny) {
    *score_tmp = *score_tmp > 1 ? *score_tmp : 1;
  } else {
    *score_tmp = *score_tmp > 6 ? *score_tmp : 6;
  }
}

void XPUStaticKernelPickPass::NodeOutputPrecision(
    const std::unique_ptr<SSAGraph>& graph, lite::mir::Node* node) {
  auto& inst = node->AsStmt();
  if (inst.op_type() == "fetch") {
    return;
  }

  const auto* op_info = inst.op_info();
  for (auto* out_node : node->outlinks) {
    auto& var = out_node->AsArg();
    const auto& var_name = var.name;
    std::string arg_name;
    CHECK(op_info->GetOutputArgname(var_name, &arg_name))
        << "Can not find the output argument,current var name : " << var_name;
    VLOG(6) << " output arg name:" << arg_name << " var name:" << var_name;
    Scope* scope = node->AsStmt().op()->scope();
    auto* var_ptr = scope->FindVar(var_name);
    if (var_ptr == nullptr) {
      VLOG(6) << "Can't find ouput var_name:  " << var_name
              << "in current scope.";
      continue;
    }
    if (!var_ptr->IsType<lite::Tensor>()) {
      continue;
    }

    PrecisionType precison = var_ptr->GetMutable<lite::Tensor>()->precision();
    xpu_output_type_.emplace(var_name, precison);
  }
}

void XPUStaticKernelPickPass::SpecialNodeOutputPrecision(
    const std::unique_ptr<SSAGraph>& graph,
    lite::mir::Node* node,
    const std::unique_ptr<lite::KernelBase>& kernel) {
  auto& inst = node->AsStmt();

  std::vector<std::string> out_var_names;
  const auto* op_info = inst.op_info();
  for (auto* out_node : node->outlinks) {
    CHECK(out_node->IsArg());
    auto& var = out_node->AsArg();
    const auto& var_name = var.name;
    std::string arg_name;

    CHECK(op_info->GetOutputArgname(var_name, &arg_name))
        << "Can not find the output argument, current var name : " << var_name;
    VLOG(6) << " output arg name:" << arg_name << " var name:" << var_name;
    if (out_node->outlinks.empty()) {
      continue;
    }

    const auto* decl_type = kernel->GetOutputDeclType(arg_name);
    CHECK(decl_type);
    PrecisionType precison = decl_type->precision();
    xpu_output_type_.emplace(var_name, precison);
  }
}

void XPUStaticKernelPickPass::InplaceNodeOutputPrecision(
    lite::mir::Node* node) {
  PrecisionType pre_op_output_precision = PrecisionType::kUnk;
  auto& instruct = node->AsStmt();
  for (auto* in_var_node : node->inlinks) {
    CHECK(in_var_node->IsArg());
    auto& var = in_var_node->AsArg();
    const auto& var_name = var.name;

    std::string tmp;
    CHECK(instruct.op_info()->GetInputArgname(var_name, &tmp));
    VLOG(6) << "current kernel input data variable name:" << var_name
            << "Parameter name:" << tmp;
    if (in_var_node->inlinks.empty()) {
      continue;
    }

    if (xpu_output_type_.count(var_name)) {
      pre_op_output_precision = xpu_output_type_[var_name];
    }
  }

  // collect inplace op output data precision
  if (pre_op_output_precision != PrecisionType::kUnk) {
    for (auto* out_var_node : node->outlinks) {
      CHECK(out_var_node->IsArg());
      auto& var = out_var_node->AsArg();
      const auto& var_name = var.name;
      std::string tmp;
      CHECK(instruct.op_info()->GetOutputArgname(var_name, &tmp));
      if (out_var_node->outlinks.empty()) {
        continue;
      }

      xpu_output_type_.emplace(var_name, pre_op_output_precision);
    }
  }
}

// Special nodes like conv2d, matmul ; collect input data precision for eatch
// registry kernel as a candidate set.
void XPUStaticKernelPickPass::SpecialNodeInputPrecision(lite::mir::Node* node,
                                                        const bool collect_int8,
                                                        const bool collect_fp16,
                                                        bool* has_collected) {
  auto& inst = node->AsStmt();
  const auto* op_info = inst.op_info();
  if (collect_int8) {
    if (!node->AsStmt().mutable_op_info()->HasAttr("enable_int8")) {
      return;
    }

    if (!node->AsStmt().mutable_op_info()->GetAttr<bool>("enable_int8")) {
      return;
    }
  }

  for (auto* in_var_node : node->inlinks) {
    CHECK(in_var_node->IsArg());
    auto& var = in_var_node->AsArg();
    const auto& var_name = var.name;
    std::string arg_name;
    CHECK(op_info->GetInputArgname(var_name, &arg_name))
        << "Can not find the input argument,current var name : " << var_name;
    VLOG(6) << " input arg name:" << arg_name << " var name:" << var_name;
    if (in_var_node->inlinks.empty()) {
      VLOG(6) << "Ignore input arg name:" << arg_name
              << ",var name:" << var_name;
      continue;
    }

    std::vector<std::map<std::string, PrecisionType>> kernel_input_type{};
    for (auto&& kernel : inst.kernels()) {
      if (collect_int8 && kernel->precision() != PrecisionType::kInt8) {
        continue;
      }

      if (collect_fp16 && kernel->precision() == PrecisionType::kInt8) {
        continue;
      }

      if (kernel->summary().find(xpu_disable_flag_) != std::string::npos) {
        VLOG(6) << " ignore collect current kernel:" << kernel->summary();
        continue;
      }

      std::map<std::string, PrecisionType> tmp_map;
      PrecisionType precison;

      const auto* decl_type = kernel->GetInputDeclType(arg_name);
      CHECK(decl_type);
      precison = decl_type->precision();
      tmp_map.emplace(kernel->summary(), precison);
      kernel_input_type.emplace_back(std::move(tmp_map));
    }

    xpu_input_type_.emplace(var_name, kernel_input_type);
  }
  *has_collected = true;
}

void XPUStaticKernelPickPass::NodeInputPrecision(
    lite::mir::Node* node, const std::unique_ptr<SSAGraph>& graph) {
  auto& inst = node->AsStmt();
  if (inst.op_type() == "feed") {
    return;
  }

  const auto* op_info = inst.op_info();
  for (auto* in_node : node->inlinks) {
    CHECK(in_node->IsArg());
    auto& var = in_node->AsArg();
    const auto& var_name = var.name;
    std::string arg_name;
    CHECK(op_info->GetInputArgname(var_name, &arg_name))
        << "Can not find the input argument,current var name : " << var_name;
    VLOG(6) << " input arg name:" << arg_name << " var name:" << var_name;

    std::vector<std::map<std::string, PrecisionType>> kernel_input_type{};
    std::map<std::string, PrecisionType> tmp_map;
    PrecisionType precison;
    Scope* scope = node->AsStmt().op()->scope();

    auto* var_ptr = scope->FindVar(var_name);
    if (var_ptr == nullptr) {
      VLOG(6) << "Can't find input var_name:  " << var_name
              << "in current scope.";
      continue;
    }

    if (!var_ptr->IsType<lite::Tensor>()) {
      continue;
    }

    precison = var_ptr->GetMutable<lite::Tensor>()->precision();
    tmp_map.emplace(inst.op_type(), precison);
    kernel_input_type.emplace_back(std::move(tmp_map));
    VLOG(6) << "var name:" << var_name << "inst.op_type():" << inst.op_type()
            << static_cast<int>(precison);
    xpu_input_type_.emplace(var_name, kernel_input_type);
  }
}

// Special for inplace op.
void XPUStaticKernelPickPass::InplaceNodeInputPrecision(lite::mir::Node* node) {
  auto& inst = node->AsStmt();
  const auto* op_info = inst.op_info();
  // inplace op only has one inpute variable.
  std::string inplace_op_input_name{"none"};
  for (auto* in_node : node->inlinks) {
    CHECK(in_node->IsArg());
    auto& var = in_node->AsArg();
    const auto& var_name = var.name;
    std::string arg_name;
    CHECK(op_info->GetInputArgname(var_name, &arg_name))
        << "Can not find the input argument,current var name : " << var_name;
    VLOG(6) << " input arg name:" << arg_name << " var name:" << var_name;
    if (in_node->inlinks.empty()) {
      continue;
    }

    inplace_op_input_name = var_name;
  }

  for (auto* out_node : node->outlinks) {
    CHECK(out_node->IsArg());
    auto& var = out_node->AsArg();
    const auto& var_name = var.name;
    std::string arg_name;
    int num = 0;

    CHECK(op_info->GetOutputArgname(var_name, &arg_name))
        << "Can not find the output argument,current var name : " << var_name;
    VLOG(6) << " output arg name:" << arg_name << " var name:" << var_name;
    // inplace op only have one output variable,but ic can connect input
    // variables of multiple Ops
    int output_match_num = xpu_input_type_.count(var_name);
    if (out_node->outlinks.empty() || output_match_num == 0) {
      continue;
    }

    for (auto iter = xpu_input_type_.begin(); iter != xpu_input_type_.end();
         ++iter) {
      if (num >= output_match_num) {
        break;
      }

      if (iter->first != var_name) {
        continue;
      }

      ++num;
      xpu_input_type_.emplace(inplace_op_input_name, iter->second);
    }
    VLOG(6) << "inplace op :" << inst.op_type() << "input prision"
            << "replace by the next op input prision ";
    VLOG(6) << "inplace op :" << inst.op_type()
            << ", inpute name:" << inplace_op_input_name
            << ",the next op input input name : " << var_name;
  }
}

void XPUStaticKernelPickPass::InplaceOpScore(lite::mir::Node* node,
                                             const lite::KernelBase& kernel,
                                             bool* type_match,
                                             size_t* score) {
  auto& instruct = node->AsStmt();
  PrecisionType pre_op_output_precision = PrecisionType::kUnk;
  for (auto* in_node : node->inlinks) {
    CHECK(in_node->IsArg());
    auto& var = in_node->AsArg();
    const auto& var_name = var.name;
    std::string tmp;

    CHECK(instruct.op_info()->GetInputArgname(var_name, &tmp));
    VLOG(6) << "current kernel input data variable name:" << var_name
            << "Parameter name:" << tmp;
    if (in_node->inlinks.empty() && xpu_output_type_.count(var_name) == 0) {
      continue;
    }

    // only to match input X
    if (tmp != "X") {
      continue;
    }

    if (xpu_output_type_.count(var_name)) {
      size_t score_tmp = 0;
      pre_op_output_precision = xpu_output_type_[var_name];
      if (kernel.GetInputDeclType(tmp)->precision() == PrecisionType::kAny) {
        GetScore(PrecisionType::kAny, &score_tmp);
        VLOG(6) << "current inplace kernel input data precision:kAny";
      }

      if (pre_op_output_precision ==
              kernel.GetInputDeclType(tmp)->precision() ||
          pre_op_output_precision == PrecisionType::kAny) {
        GetScore(pre_op_output_precision, &score_tmp);
        *type_match = true;
        VLOG(6) << "inplace op match input data precision";
      }

      *score += score_tmp;
    }
  }

  // collect inplace op output data precision
  if (pre_op_output_precision != PrecisionType::kUnk) {
    for (auto* out_node : node->outlinks) {
      CHECK(out_node->IsArg());
      auto& var = out_node->AsArg();
      const auto& var_name = var.name;
      std::string tmp;
      CHECK(instruct.op_info()->GetOutputArgname(var_name, &tmp));
      if (out_node->outlinks.empty() && xpu_input_type_.count(var_name) == 0) {
        continue;
      }

      xpu_output_type_.emplace(var_name, pre_op_output_precision);
    }
  }
}

void XPUStaticKernelPickPass::SpecialOpScore(lite::mir::Node* node,
                                             const lite::KernelBase& kernel,
                                             bool* type_match,
                                             size_t* score) {
  size_t score_tmp_all = 0;
  bool intput_match = true;
  bool output_match = true;
  bool consider_cpu = false;

  auto& instruct = node->AsStmt();

  if (consider_cpu_op_.count(instruct.op_type())) {
    consider_cpu = true;
  }

  if (!(kernel.place().target == TARGET(kXPU) || consider_cpu)) {
    return;
  }

  // type cast bug，We temporarily add it here
  for (auto* in_node : node->inlinks) {
    CHECK(in_node->IsArg());
    auto& var = in_node->AsArg();
    const auto& var_name = var.name;
    std::string tmp;
    CHECK(instruct.op_info()->GetInputArgname(var_name, &tmp));
    if (in_node->inlinks.empty() && xpu_output_type_.count(var_name) == 0) {
      if (kernel.GetInputDeclType(tmp)->precision() == PrecisionType::kFP16) {
        *score = 0;
        VLOG(6) << "not pick fp16 kernel ,because  input weight "
                   "is not fp16.";
        return;
      }
    }
  }

  // input data precision score
  for (auto* in_node : node->inlinks) {
    CHECK(in_node->IsArg());
    auto& var = in_node->AsArg();
    const auto& var_name = var.name;
    std::string tmp;
    CHECK(instruct.op_info()->GetInputArgname(var_name, &tmp));
    if (in_node->inlinks.empty() && xpu_output_type_.count(var_name) == 0) {
      continue;
    }

    if (tmp == "Branch") {
      continue;
    }

    if (xpu_output_type_.count(var_name) == 0) {
      continue;
    }

    VLOG(6) << "current kernel input data variable name:" << var_name
            << ", Parameter name:" << tmp;

    size_t score_tmp = 0;
    if (kernel.GetInputDeclType(tmp)->precision() == PrecisionType::kAny) {
      GetScore(PrecisionType::kAny, &score_tmp);
      VLOG(6) << "match input data precision:kAny";
    }

    if (xpu_output_type_[var_name] ==
            kernel.GetInputDeclType(tmp)->precision() ||
        xpu_output_type_[var_name] == PrecisionType::kAny) {
      GetScore(xpu_output_type_[var_name], &score_tmp);
      VLOG(6) << "match input data precision";
    }

    if (score_tmp == 0) {
      intput_match = false;
    }

    score_tmp_all += score_tmp;
  }

  // output data precision score
  for (auto* out_node : node->outlinks) {
    CHECK(out_node->IsArg());
    auto& var = out_node->AsArg();
    const auto& var_name = var.name;
    std::string tmp;
    CHECK(instruct.op_info()->GetOutputArgname(var_name, &tmp));
    int output_match_num = xpu_input_type_.count(var_name);
    if (out_node->outlinks.empty() && output_match_num == 0) {
      continue;
    }

    if (output_match_num == 0) {
      continue;
    }

    VLOG(6) << "current kernel output data variable name:" << var_name
            << ", Parameter name:" << tmp;
    int num = 0;
    size_t score_tmp = 0;
    for (auto iter = xpu_input_type_.begin(); iter != xpu_input_type_.end();
         ++iter) {
      if (num >= output_match_num) {
        break;
      }

      if (iter->first != var_name) {
        continue;
      }

      ++num;
      for (auto& map_kernel : iter->second) {
        // Special op fetch
        if (map_kernel.begin()->first.substr(0, 5) == "fetch") {
          if (map_kernel.begin()->second ==
              kernel.GetOutputDeclType(tmp)->precision()) {
            score_tmp = 1000;
          }
          continue;
        }

        if (kernel.GetOutputDeclType(tmp)->precision() == PrecisionType::kAny) {
          VLOG(6) << "match precision:kAny,the next kernel's name:"
                  << map_kernel.begin()->first;
          GetScore(PrecisionType::kAny, &score_tmp);
        }

        if (map_kernel.begin()->second ==
                kernel.GetOutputDeclType(tmp)->precision() ||
            map_kernel.begin()->second == PrecisionType::kAny) {
          VLOG(6) << "match next kernel's input data precision,the "
                     "next kernel name:"
                  << map_kernel.begin()->first;
          GetScore(map_kernel.begin()->second, &score_tmp);
        }
      }
    }

    if (score_tmp == 0) {
      output_match = false;
    }

    score_tmp_all += score_tmp;
  }

  if (score_tmp_all > 0) {
    *type_match = intput_match & output_match;
  }

  *score += score_tmp_all;
}

void XPUStaticKernelPickPass::SliceForceNotUseXPU(
    lite::mir::Node* node,
    const lite::KernelBase& kernel,
    bool* type_match,
    size_t* score) {
  for (auto in_var_node : node->inlinks) {
    CHECK(in_var_node->IsArg());
    if (in_var_node->inlinks.empty()) continue;
    for (auto iter_node = in_var_node->inlinks.begin();
         iter_node != in_var_node->inlinks.end();
         iter_node++) {
      if (!(*iter_node)->IsStmt()) continue;
      if (((*iter_node)->AsStmt().op_type() == "shape") &&
          (kernel.place().target == TARGET(kXPU))) {
        *score = 0;
      }
    }
  }
}

void XPUStaticKernelPickPass::GeneralInt8OpScore(lite::mir::Node* node,
                                                 const lite::KernelBase& kernel,
                                                 bool* type_match,
                                                 size_t* score) {
  auto& instruct = node->AsStmt();
  for (auto* in_node : node->inlinks) {
    CHECK(in_node->IsArg());
    auto& var = in_node->AsArg();
    const auto& var_name = var.name;
    std::string tmp;
    CHECK(instruct.op_info()->GetInputArgname(var_name, &tmp));
    if (in_node->inlinks.empty() && xpu_output_type_.count(var_name) == 0) {
      continue;
    }

    if (xpu_output_type_.count(var_name) == 0) {
      continue;
    }

    VLOG(6) << "current kernel input data variable name:" << var_name
            << ", Parameter name:" << tmp;

    size_t score_tmp = 0;
    if (kernel.GetInputDeclType(tmp)->precision() == PrecisionType::kAny) {
      GetScore(PrecisionType::kAny, &score_tmp);
      VLOG(6) << "match input data precision:kAny";
    }

    if (xpu_output_type_[var_name] ==
            kernel.GetInputDeclType(tmp)->precision() ||
        xpu_output_type_[var_name] == PrecisionType::kAny) {
      GetScore(xpu_output_type_[var_name], &score_tmp);
      VLOG(6) << "match input data precision";
    }

    if (score_tmp > 0) {
      *type_match = true;
    }

    *score += score_tmp;
  }
}

void XPUStaticKernelPickPass::GradeXPUKernelScore(
    lite::mir::Node* node,
    const lite::KernelBase& kernel,
    const paddle::lite::mir::Node::Stmt& instruct,
    const std::vector<std::string>& in_names,
    const std::vector<std::string>& out_names,
    const std::map<std::string, PrecisionType>& in_types,
    const std::map<std::string, PrecisionType>& out_types,
    size_t* score,
    bool* type_match) {
  // Some registered kernel cannot be used.
  if (kernel.summary().find(xpu_disable_flag_) != std::string::npos) {
    *score = 0;
    VLOG(6) << " ignore pick current kernel:" << kernel.summary();
    return;
  }

  bool kernel_force_use = ForceUsePrecision(score, kernel, instruct);
  if (kernel_force_use) {
    VLOG(4) << "[xpu kernel force use score s4]:" << score;
    return;
  }

  // kernel compute precision:int8/int16,data precicion:int8/fp16/fp32
  if (xpu_use_fp16_optimizer_ || xpu_use_int8_optimizer_) {
    if (xpu_inplace_op_.count(instruct.op_type())) {
      InplaceOpScore(node, kernel, type_match, score);
      return;
    }

    if ((xpu_use_fp16_optimizer_ &&
         xpu_special_op_.count(instruct.op_type())) ||
        (xpu_use_int8_optimizer_ &&
         instruct.op_info()->HasAttr("enable_int8") &&
         instruct.op_info()->GetAttr<bool>("enable_int8") &&
         xpu_int8_special_op_.count(instruct.op_type()))) {
      SpecialOpScore(node, kernel, type_match, score);
      return;
    }

    if (xpu_use_int8_optimizer_ && instruct.op_info()->HasAttr("enable_int8") &&
        instruct.op_info()->GetAttr<bool>("enable_int8") &&
        xpu_int8_general_op_.count(instruct.op_type())) {
      GeneralInt8OpScore(node, kernel, type_match, score);
      return;
    }
  }

  if (instruct.op_type() == "slice") {
    SliceForceNotUseXPU(node, kernel, type_match, score);
  }

  // kernel compute precision:fp32(int16),data precicion:fp32
  if (!(instruct.op_info()->HasAttr("enable_int8") &&
        instruct.op_info()->GetAttr<bool>("enable_int8")) ||
      xpu_int8_special_op_.count(instruct.op_type()) == 0) {
    *type_match = true;
    if (instruct.op_type() == "feed") {
      for (size_t i = 0; i < out_names.size(); ++i) {
        std::string tmp;
        CHECK(instruct.op_info()->GetOutputArgname(out_names[i], &tmp));
        if (out_types.count(out_names[i]) &&
            out_types.at(out_names[i]) !=
                kernel.GetOutputDeclType(tmp)->precision()) {
          *type_match = false;
        }
      }
    } else {
      for (size_t i = 0; i < in_names.size(); ++i) {
        std::string tmp;
        CHECK(instruct.op_info()->GetInputArgname(in_names[i], &tmp));
        if (in_types.count(in_names[i])) {
          if (!PrecTypeCompatible(in_types.at(in_names[i]),
                                  kernel.GetInputDeclType(tmp)->precision())) {
            *type_match = false;
          } else {
            *score += 1;
          }
        }
      }
    }
  }

  return;
}

void XPUStaticKernelPickPass::CollectXPUSpecialOPType(
    const std::unique_ptr<SSAGraph>& graph) {
  for (auto& node : graph->StmtTopologicalOrder()) {
    if (!node->IsStmt()) continue;

    auto& instruct = node->AsStmt();
    for (auto&& kernel : instruct.kernels()) {
      if (kernel->place().target != TARGET(kXPU)) {
        continue;
      }

      auto op_type = instruct.op_type();

      if (xpu_use_fp16_optimizer_) {
        if (kernel->precision() == PrecisionType::kFP16) {
          xpu_special_op_.emplace(op_type);
          continue;
        }
      }
    }
  }

  for (auto op_type : xpu_special_op_) {
    VLOG(6) << "Collected xpu fp16 precioson op:" << op_type;
  }

  return;
}

void XPUStaticKernelPickPass::strategiesconcatOP(
    const std::unique_ptr<SSAGraph>& graph,
    lite::mir::Node* op_node,
    bool* quant_int8) {
  if (!quant_int8) {
    return;
  }
  auto op_info = op_node->AsStmt().mutable_op_info();
  float cancat_out_scale = 0;
  std::string cancat_out_var_name;
  for (auto out_var_node : op_node->outlinks) {
    CHECK(out_var_node->IsArg());
    cancat_out_var_name = out_var_node->arg()->name;
    if (op_info->HasOutputScale(cancat_out_var_name)) {
      cancat_out_scale = op_info->GetOutputScale(cancat_out_var_name)[0];
      break;
    }
  }

  if (cancat_out_scale <= 0) {
    *quant_int8 = false;
    return;
  }

  // case1.consumer of concat is those op,which is not nedd scale,so concat outs
  // scale value is Untrustworthy.
  if (xpu_int8_compute_autotune_) {
    for (auto out_var_node : op_node->outlinks) {
      CHECK(out_var_node->IsArg());
      if (out_var_node->outlinks.empty()) continue;
      for (auto iter_node = out_var_node->outlinks.begin();
           iter_node != out_var_node->outlinks.end();
           iter_node++) {
        if (!(*iter_node)->IsStmt()) continue;
        auto next_op_type = (*iter_node)->AsStmt().mutable_op_info()->Type();
        if (xpu_int8_general_op_not_need_sacale_.count(next_op_type)) {
          *quant_int8 = false;
          return;
        }
      }
    }
  }

  // case2.Producer of concat has more than one output.
  for (auto in_var_node : op_node->inlinks) {
    CHECK(in_var_node->IsArg());
    auto concat_in_var_name = in_var_node->arg()->name;
    if (in_var_node->inlinks.empty()) continue;
    for (auto iter_node = in_var_node->inlinks.begin();
         iter_node != in_var_node->inlinks.end();
         iter_node++) {
      if (!(*iter_node)->IsStmt()) continue;

      for (auto preinlik_op_out_var_node : (*iter_node)->outlinks) {
        CHECK(preinlik_op_out_var_node->IsArg());
        auto var_name = preinlik_op_out_var_node->arg()->name;
        if (var_name != concat_in_var_name) {
          continue;
        }

        if (preinlik_op_out_var_node->outlinks.size() > 1) {
          auto pre_op_info = (*iter_node)->AsStmt().mutable_op_info();
          if (pre_op_info->HasOutputScale(var_name)) {
            float pre_out_scale = pre_op_info->GetOutputScale(var_name)[0];
            if (abs(pre_out_scale - cancat_out_scale) / cancat_out_scale >
                0.1f) {
              *quant_int8 = false;
              return;
            }
          }
        }
      }
    }
  }

  // Reset concat input sclae values
  for (auto in_var_node : op_node->inlinks) {
    CHECK(in_var_node->IsArg());
    auto in_var_name = in_var_node->arg()->name;
    if (!op_info->HasInputScale(in_var_name)) continue;
    op_info->SetInputScale(in_var_name, {cancat_out_scale});
  }

  auto& cur_instruct = op_node->AsStmt();
  auto cur_update_desc = *cur_instruct.mutable_op_info();
  cur_instruct.ResetOp(cur_update_desc, graph->valid_places());

  // Reset the out scale ，which the producers of concat.
  for (auto in_var_node : op_node->inlinks) {
    CHECK(in_var_node->IsArg());
    auto concat_in_var_name = in_var_node->arg()->name;

    if (in_var_node->inlinks.empty()) continue;
    for (auto iter_node = in_var_node->inlinks.begin();
         iter_node != in_var_node->inlinks.end();
         iter_node++) {
      if (!(*iter_node)->IsStmt()) continue;
      auto pre_op_type = (*iter_node)->AsStmt().mutable_op_info()->Type();
      // Reset pre-pre inlinks op output sclae value,if pre inlinks op is
      // inplace.
      if (xpu_inplace_op_.count(pre_op_type) ||
          xpu_int8_general_op_not_need_sacale_.count(pre_op_type)) {
        auto inplace_input_var_node = (*iter_node)->inlinks.front();
        CHECK(inplace_input_var_node->IsArg());
        auto inplace_input_var_name = inplace_input_var_node->arg()->name;
        auto pre_pre_node = inplace_input_var_node->inlinks.front();
        if (!pre_pre_node->IsStmt()) {
          continue;
        }

        auto pre_pre_op_info = pre_pre_node->AsStmt().mutable_op_info();
        auto& pre_pre_instruct = pre_pre_node->AsStmt();

        for (auto pre_pre_op_out_var_node : pre_pre_node->outlinks) {
          CHECK(pre_pre_op_out_var_node->IsArg());
          auto var_name = pre_pre_op_out_var_node->arg()->name;
          if (inplace_input_var_name != var_name) {
            continue;
          }

          if (pre_pre_op_info->HasOutputScale(var_name)) {
            VLOG(4) << "OP type : " << pre_pre_op_info->Type()
                    << ", origin out sacle is:"
                    << pre_pre_op_info->GetOutputScale(var_name)[0]
                    << ", Reset out scale is :" << cancat_out_scale;
            pre_pre_op_info->SetOutputScale(var_name, {cancat_out_scale});
            break;
          }
        }

        auto pre_pre_update_desc = *pre_pre_instruct.mutable_op_info();
        pre_pre_instruct.ResetOp(pre_pre_update_desc, graph->valid_places());
      }

      // Reset pre inlinks op output sclae value
      auto pre_op_info = (*iter_node)->AsStmt().mutable_op_info();
      auto& pre_instruct = (*iter_node)->AsStmt();

      for (auto preinlik_op_out_var_node : (*iter_node)->outlinks) {
        CHECK(preinlik_op_out_var_node->IsArg());
        auto out_var_name = preinlik_op_out_var_node->arg()->name;
        if (concat_in_var_name != out_var_name) {
          continue;
        }

        if (pre_op_info->HasOutputScale(out_var_name)) {
          VLOG(4) << "OP type : " << pre_op_info->Type()
                  << ", origin out sacle is:"
                  << pre_op_info->GetOutputScale(out_var_name)[0]
                  << ", Reset out scale is :" << cancat_out_scale;
          pre_op_info->SetOutputScale(out_var_name, {cancat_out_scale});
          break;
        }
      }
      auto pre_update_desc = *pre_instruct.mutable_op_info();
      pre_instruct.ResetOp(pre_update_desc, graph->valid_places());
    }
  }

  if (xpu_int8_compute_autotune_) {
    for (auto in_var_node : op_node->inlinks) {
      CHECK(in_var_node->IsArg());
      auto in_var_name = in_var_node->arg()->name;
      if (op_info->HasInputScale(in_var_name)) {
        float input_scale = op_info->GetInputScale(in_var_name)[0];
        if (abs(input_scale - cancat_out_scale) > 0.01f) {
          *quant_int8 = false;
          break;
        }
      }
    }
  }
}

// Only pick some ops for  int8 compute in XPU.
// Op pick int8 kernel in XPU, when has enable_int8 attribute.
void XPUStaticKernelPickPass::strategiesInt8OP(lite::mir::Node* op_node,
                                               bool* quant_int8) {
  auto& instruct = op_node->AsStmt();
  auto op_type = instruct.mutable_op_info()->Type();
  // Use some strategies to evaluate whether the current OP can use int8
  // compute.

  // 1.The current op uses int8 to compute without relying on scale.
  if (xpu_int8_general_op_not_need_sacale_.count(op_type)) {
    *quant_int8 = true;
    return;
  }

  // 2.The next consumers OP must have attribute enable_int8.
  *quant_int8 = false;
  for (auto out_var_node : op_node->outlinks) {
    CHECK(out_var_node->IsArg());
    if (out_var_node->outlinks.empty()) continue;
    for (auto iter_node = out_var_node->outlinks.begin();
         iter_node != out_var_node->outlinks.end();
         iter_node++) {
      if (!(*iter_node)->IsStmt()) continue;
      auto next_op_info = (*iter_node)->AsStmt().mutable_op_info();
      if (next_op_info->HasAttr("enable_int8") &&
          next_op_info->GetAttr<bool>("enable_int8")) {
        *quant_int8 = true;
        break;
      }
    }
  }

  // Only eatch producer op has attribute enable_int8,consumer op can
  // set the attribute of enable_int8 to true;
  if (op_node->inlinks.empty()) {
    *quant_int8 = false;
    return;
  }
  for (auto in_var_node : op_node->inlinks) {
    CHECK(in_var_node->IsArg());
    if (in_var_node->inlinks.empty()) continue;
    for (auto iter_node = in_var_node->inlinks.begin();
         iter_node != in_var_node->inlinks.end();
         iter_node++) {
      if (!(*iter_node)->IsStmt()) continue;
      auto pre_op_info = (*iter_node)->AsStmt().mutable_op_info();
      if (!pre_op_info->HasAttr("enable_int8")) {
        *quant_int8 = false;
        break;
      }
      if (!pre_op_info->GetAttr<bool>("enable_int8")) {
        *quant_int8 = false;
        break;
      }
    }
  }

  if (*quant_int8) {
    return;
  }

  // 3.The relative threshold between the input maximum and output maximum
  // of the current op is less than 10%.
  *quant_int8 = true;
  float out_scale = 0.0f;
  if (instruct.mutable_op_info()->HasAttr("Out0_scale")) {
    out_scale = instruct.mutable_op_info()->GetAttr<std::vector<float>>(
        "Out0_scale")[0];
  } else {
    *quant_int8 = false;
  }

  for (auto in_var_node : op_node->inlinks) {
    CHECK(in_var_node->IsArg());
    auto in_var_name = in_var_node->arg()->name;
    if (instruct.mutable_op_info()->HasInputScale(in_var_name)) {
      float input_scale =
          instruct.mutable_op_info()->GetInputScale(in_var_name)[0];
      if ((abs(input_scale - out_scale) / out_scale) > 0.1f) {
        *quant_int8 = false;
        break;
      }
    } else {
      *quant_int8 = false;
    }
  }
  if (*quant_int8) {
    return;
  }

  // 4.special op ew_add
  if (op_type == "elementwise_add") {
    out_scale = 0.0f;
    if (instruct.mutable_op_info()->HasAttr("Out0_scale")) {
      out_scale = instruct.mutable_op_info()->GetAttr<std::vector<float>>(
          "Out0_scale")[0];
    }

    for (auto in_var_node : op_node->inlinks) {
      CHECK(in_var_node->IsArg());
      auto in_var_name = in_var_node->arg()->name;
      if (instruct.mutable_op_info()->HasInputScale(in_var_name)) {
        float input_scale =
            instruct.mutable_op_info()->GetInputScale(in_var_name)[0];
        if (abs((input_scale - out_scale) / out_scale) < 0.1f) {
          *quant_int8 = true;
          return;
        }
      }
    }
  }

  // Not satisfied  all current conditions.
  *quant_int8 = false;
}

void XPUStaticKernelPickPass::SetEnableInt8Attribute(
    const std::unique_ptr<SSAGraph>& graph) {
  for (auto& op_node : graph->StmtTopologicalOrder()) {
    if (!op_node->IsStmt()) continue;
    auto& instruct = op_node->AsStmt();
    auto op_type = instruct.mutable_op_info()->Type();
    if (!(xpu_int8_general_op_.count(op_type) ||
          xpu_inplace_op_.count(op_type))) {
      continue;
    }

    bool quant_int8 = true;
    // wing to slim bug,Temp add.
    // Decide the enable_int8 attribute of general int8 op is true or false.
    if (xpu_int8_general_op_.count(op_type)) {
      for (auto in_var_node : op_node->inlinks) {
        CHECK(in_var_node->IsArg());
        auto in_var_name = in_var_node->arg()->name;
        if (!instruct.mutable_op_info()->HasInputScale(in_var_name)) {
          quant_int8 = false;
          break;
        }
      }

      // Owing to slim bug,this stategy added for ppyolo.
      if (op_type == "concat" || op_type == "split") {
        for (auto out_var_node : op_node->outlinks) {
          CHECK(out_var_node->IsArg());
          auto out_var_name = out_var_node->arg()->name;
          if (!instruct.mutable_op_info()->HasOutputScale(out_var_name)) {
            quant_int8 = false;
            break;
          }
        }
      }
    }

    // At least one producer's int8 attribute is true.
    {
      bool enable_int8 = false;
      for (auto in_var_node : op_node->inlinks) {
        CHECK(in_var_node->IsArg());
        if (in_var_node->inlinks.empty()) continue;
        for (auto iter_node = in_var_node->inlinks.begin();
             iter_node != in_var_node->inlinks.end();
             iter_node++) {
          if (!(*iter_node)->IsStmt()) continue;
          auto pre_op_info = (*iter_node)->AsStmt().mutable_op_info();
          if (pre_op_info->HasAttr("enable_int8") &&
              pre_op_info->GetAttr<bool>("enable_int8")) {
            enable_int8 = true;
            break;
          }
        }
      }
      if (!enable_int8) {
        quant_int8 = false;
      }
    }

    if (!quant_int8) {
      instruct.mutable_op_info()->SetAttr<bool>("enable_int8", false);
      auto update_desc = *instruct.mutable_op_info();
      instruct.ResetOp(update_desc, graph->valid_places());
      continue;
    }

    // hard code
    {
      quant_int8 = false;
      std::queue<lite::mir::Node*> next_ops{};
      next_ops.push(op_node);
      int num = 0;
      lite::mir::Node* cur_node;
      while (num < 5) {
        if (next_ops.size() > 0) {
          cur_node = next_ops.front();
          next_ops.pop();
        }

        for (auto out_var_node : cur_node->outlinks) {
          CHECK(out_var_node->IsArg());
          if (out_var_node->outlinks.empty()) continue;
          for (auto iter_node = out_var_node->outlinks.begin();
               iter_node != out_var_node->outlinks.end();
               iter_node++) {
            if (!(*iter_node)->IsStmt()) continue;
            auto op_type = (*iter_node)->AsStmt().mutable_op_info()->Type();
            if (xpu_int8_special_op_.count(op_type)) {
              quant_int8 = true;
              break;
            }
            next_ops.push((*iter_node));
          }
        }
        if (quant_int8) {
          break;
        }
        if (xpu_inplace_op_.count(
                cur_node->AsStmt().mutable_op_info()->Type())) {
          continue;
        }
        num++;
      }

      if (!quant_int8) {
        instruct.mutable_op_info()->SetAttr<bool>("enable_int8", false);
        auto update_desc = *instruct.mutable_op_info();
        instruct.ResetOp(update_desc, graph->valid_places());
        continue;
      }
    }

    if (xpu_int8_compute_autotune_) {
      strategiesInt8OP(op_node, &quant_int8);
    }

    // when quant op is concat,the input and output values must be the same.
    if (op_type == "concat" && quant_int8) {
      strategiesconcatOP(graph, op_node, &quant_int8);
    }

    if (!quant_int8) {
      instruct.mutable_op_info()->SetAttr<bool>("enable_int8", false);
    } else {
      instruct.mutable_op_info()->SetAttr<bool>("enable_int8", true);
    }
    VLOG(4) << "op_type:" << op_type << "quant_int8:" << quant_int8;
    auto update_desc = *instruct.mutable_op_info();
    instruct.ResetOp(update_desc, graph->valid_places());
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__static_kernel_pick_pass,
                  paddle::lite::mir::XPUStaticKernelPickPass)
    .BindTargets({TARGET(kXPU)});
