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
#include <list>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#ifdef LITE_WITH_XPU
#include "lite/backends/xpu/target_wrapper.h"
#endif
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
  kernel_pick_factors_.ConsiderTarget();
  kernel_pick_factors_.ConsiderPrecision();
  kernel_pick_factors_.ConsiderDataLayout();
  CHECK(kernel_pick_factors_.any_factor_considered())
      << "kernel_pick_factors should be specified first";
  CHECK(graph) << "graph not valid";

// Collect input data precision for each node in the graph
// Collect XPU op type,which used in fp16/in8;
#ifdef LITE_WITH_XPU
  DataPrecisionDicide(graph);
  GetXPUDeviceType();
  if (xpu_use_fp16_optimizer_ || xpu_use_int8_optimizer_) {
    CollectXPUSpecialOPType(graph);
    for (auto& node : graph->StmtTopologicalOrder()) {
      if (!node->IsStmt()) continue;

      if ((xpu_use_fp16_optimizer_ &&
           xpu_special_op_.count(node->AsStmt().op_type())) ||
          (xpu_use_int8_optimizer_ &&
           xpu_int8_special_op_.count(node->AsStmt().op_type()))) {
        SpecialNodeInputPrecision(node);
        continue;
      }

      if (xpu_inplace_op_.count(node->AsStmt().op_type())) {
        continue;
      }

      NodeInputPrecision(node, graph);
    }

    for (auto& node : graph->StmtTopologicalOrder()) {
      if (!node->IsStmt()) continue;
      if (xpu_inplace_op_.count(node->AsStmt().op_type()) == 0) {
        continue;
      }

      InplaceNodeInputPrecision(node);
    }
  }
#endif

#ifdef LITE_WITH_XPU
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
    // threse precision info store in __model__ file, if selected fp16 kernel,
    // the output precision should be changed
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
      if (instruct.op_info()->HasAttr("enable_int8") &&
          instruct.op_info()->GetAttr<bool>("enable_int8") &&
          kernel->precision() != PrecisionType::kInt8 &&
          instruct.op_type() != "__xpu__multi_encoder") {
        VLOG(6) << "Ignore current kernel: " << kernel->summary()
                << ", because we only want to pick int8 precision kernel.";
        continue;
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
           xpu_int8_special_op_.count(node->AsStmt().op_type()))) {
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
#endif
}

#ifdef LITE_WITH_XPU
void XPUStaticKernelPickPass::DataPrecisionDicide(
    const std::unique_ptr<SSAGraph>& graph) {
  if (GetStringFromEnv("XPUForceUseFP16", "false") == "true") {
    xpu_use_fp16_optimizer_ = false;
    VLOG(2) << "XPU force use data precision: FP16 ";
    return;
  }

  if (graph->valid_places()[0].precision == PrecisionType::kFP16 &&
      graph->valid_places()[0].target == TargetType::kXPU) {
    xpu_use_fp16_optimizer_ = true;
    VLOG(2) << "XPU auto use data precision: FP16/FP32/INT16 ";
    return;
  }

  if (graph->valid_places()[0].precision == PrecisionType::kInt8 &&
      graph->valid_places()[0].target == TargetType::kXPU) {
    xpu_use_int8_optimizer_ = true;
    VLOG(2) << "XPU auto use data precision: FP16/FP32/INT16/INT8 ";

    return;
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
  bool xpu_local_quant =
      GetBoolFromEnv("XPU_LOCAL_QUANT") || lite::TargetWrapperXPU::local_quant;

  if (instruct.op_type() == "__xpu__fc") {
    if (int8_quant && kernel.alias() == "XPU_Int8_FP32_FP32") {
      *score *= 4;
      VLOG(6) << "__xpu__fc: force use PRECISON INT8: *4";
      return true;
    } else if (int16_quant && kernel.alias() == "XPUFC_INT16_FP32_FP32") {
      *score *= 4;
      VLOG(6) << "__xpu__fc: force use PRECISON INT16: *4";
      return true;
    } else if (xpu_local_quant && kernel.alias() == "XPU_FP32_LOCAL_QUANT") {
      *score *= 4;
      VLOG(6) << "__xpu__fc: force use LOCAL QUANT: *4";
      return true;
    } else if ((GetStringFromEnv("XPU_ENCODER_PRECISION", "int16") == "int31" ||
                lite::TargetWrapperXPU::multi_encoder_precision == "int31") &&
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
    *score_tmp = *score_tmp > 5 ? *score_tmp : 5;
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
void XPUStaticKernelPickPass::SpecialNodeInputPrecision(lite::mir::Node* node) {
  auto& inst = node->AsStmt();
  const auto* op_info = inst.op_info();
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
      if (xpu_use_int8_optimizer_ &&
          kernel->precision() != PrecisionType::kInt8) {
        continue;
      }

      if (xpu_use_fp16_optimizer_ &&
          kernel->precision() == PrecisionType::kInt8) {
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
            score_tmp = 500;
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

void XPUStaticKernelPickPass::GetXPUDeviceType() {
  int cur_dev_idx = 0;
  uint64_t cur_dev_attr = 0;

  XPU_CALL(xpu_current_device(&cur_dev_idx));
  XPU_CALL(xpu_device_get_attr(&cur_dev_attr, XPUATTR_MODEL, cur_dev_idx));
  if (cur_dev_attr <= 1) {
    VLOG(4) << "Currents XPU device : XPU1";
    xpu_disable_flag_ = "DISABLE_XPU1";
  } else if (cur_dev_attr >= 2 && cur_dev_attr <= 299) {
    VLOG(4) << "Currents XPU device : XPU2";
    xpu_disable_flag_ = "DISABLE_XPU2";
  } else if (cur_dev_attr >= 300 && cur_dev_attr <= 599) {
    VLOG(4) << "Currents XPU device : XPU3";
    xpu_disable_flag_ = "DISABLE_XPU3";
  } else {
    VLOG(4) << "invaid XPU device";
    xpu_disable_flag_ = "NONE";
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
         xpu_int8_special_op_.count(instruct.op_type()))) {
      SpecialOpScore(node, kernel, type_match, score);
      return;
    }
  }

  // kernel compute precision:fp32(int16),data precicion:fp32
  if (!instruct.op_info()->HasAttr("enable_int8") ||
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

#endif
}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(__xpu__static_kernel_pick_pass,
                  paddle::lite::mir::XPUStaticKernelPickPass)
    .BindTargets({TARGET(kXPU)});
