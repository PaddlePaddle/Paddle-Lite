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

#include "lite/core/optimizer/mir/opencl_memory_object_config_pass.h"
#include <list>
#include <memory>
#include "lite/core/optimizer/mir/pass_registry.h"

namespace paddle {
namespace lite {
namespace mir {

void OpenCLMemoryObjectConfigPass::Apply(
    const std::unique_ptr<SSAGraph>& graph) {
  MemoryObjectConfig(graph.get());
  CorrectArgumentPlace(graph.get());
}

std::string OpenCLMemoryObjectConfigPass::ReadMemoryObjectConfigsFromEnv() {
  std::string configs;
  auto path = GetStringFromEnv(OPENCL_MEMORY_CONFIG_FILE);
  if (!path.empty()) {
    std::vector<char> buffer;
    if (ReadFile(path, &buffer, false)) {
      if (!buffer.empty()) {
        configs.insert(configs.begin(), buffer.begin(), buffer.end());
      }
    } else {
      LOG(WARNING) << "Missing the opencl memory configuration file " << path;
    }
  }
  return configs;
}

std::set<Node*> OpenCLMemoryObjectConfigPass::GetNodesFromOpenCLOpConfig(
    SSAGraph* graph, const std::string& ocl_op_configs) {
  // Get the buffer nodes from the opencl memory configurations
  std::set<Node*> buffer_nodes;
  std::vector<std::string> lines = Split(ocl_op_configs, "\n");
  for (const auto& line : lines) {
    if (line.empty()) continue;
    std::vector<std::string> node_info = Split(line, ":");
    std::string op_type = node_info.at(0);
    std::vector<std::string> in_vars_name;
    if (node_info.size() > 1) {
      in_vars_name = Split(node_info.at(1), ",");
    }
    std::vector<std::string> out_vars_name;
    if (node_info.size() > 2) {
      out_vars_name = Split(node_info.at(2), ",");
    }

    for (auto& node : graph->mutable_nodes()) {
      if (node.IsArg()) continue;
      auto stmt = node.stmt();
      if (op_type != stmt->op_type()) continue;
      auto in_nodes = node.inlinks;
      auto out_nodes = node.outlinks;
      if (in_vars_name.size() > in_nodes.size() ||
          out_vars_name.size() > out_nodes.size()) {
        continue;
      }

      bool matched = true;
      for (auto in_var_name : in_vars_name) {
        bool find_var = false;
        for (auto* in_node : in_nodes) {
          if (in_node->arg()->name == in_var_name) {
            find_var = true;
            break;
          }
        }
        if (!find_var) {
          matched = false;
          break;
        }
      }

      for (auto out_var_name : out_vars_name) {
        bool find_var = false;
        for (auto* out_node : out_nodes) {
          if (out_node->arg()->name == out_var_name) {
            find_var = true;
            break;
          }
        }
        if (!find_var) {
          matched = false;
          break;
        }
      }

      if (matched) {
        buffer_nodes.insert(&node);
      }
    }
  }
  return buffer_nodes;
}

void OpenCLMemoryObjectConfigPass::CorrectArgumentPlace(SSAGraph* graph) {
  for (auto& x : graph->StmtTopologicalOrder()) {
    auto& inst = x->AsStmt();
    // deal with inputs
    VLOG(4) << "checking op " << inst.op_info()->Repr();

    std::string op_type = inst.op_type();
    const auto op = inst.op();
    const auto* op_info = inst.op_info();
    auto* scope = op->scope();
    bool change_image2d_to_buffer = false;
    bool change_image2d_to_cpu = false;

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

    // 1. op unsupport persistable
    const std::vector<std::string> op_cases_unsupport_persistable{
        "elementwise_floordiv",
        "elementwise_add",
        "elementwise_sub",
        "elementwise_mul",
        "elementwise_div",
        "elementwise_pow",
        "elementwise_mod",
        "reshape2",
        "unsqueeze2",
        "split"};
    if (std::find(op_cases_unsupport_persistable.begin(),
                  op_cases_unsupport_persistable.end(),
                  op_type) != op_cases_unsupport_persistable.end()) {
      for (std::list<Node*>::iterator i = x->inlinks.begin();
           i != x->inlinks.end();
           ++i) {
        std::string in_name =
            get_argname((*i)->AsArg().name, inst.op_info()->inputs());
        if (in_name == "X") {
          auto* var = scope->FindVar((*i)->arg()->name);
          if (var->Get<Tensor>().persistable()) change_image2d_to_cpu = true;
        }
      }
    }

    if (op_type == "slice") {
      for (std::list<Node*>::iterator i = x->inlinks.begin();
           i != x->inlinks.end();
           ++i) {
        std::string in_name =
            get_argname((*i)->AsArg().name, inst.op_info()->inputs());
        if (in_name == "Input") {
          auto* var = scope->FindVar((*i)->arg()->name);
          if (var->Get<Tensor>().persistable()) change_image2d_to_cpu = true;
        }
      }
    }
    if (inst.place().layout == DATALAYOUT(kImageDefault) ||
        inst.place().layout == DATALAYOUT(kImageFolder)) {
      // 2. image2d unsupport dims.size() > 4
      // 3. if input_shape_default_, according CL_DEVICE_IMAGE2D_MAX_WIDTH
      // change target
      for (auto& var_name : op_info->output_names()) {
        auto* var = scope->FindVar(var_name);
        CHECK(var) << "no variable called " << var_name << " found";
        const auto& tensor = var->Get<Tensor>();
        const auto dims = tensor.dims();
        std::string in_name = get_argname(var_name, inst.op_info()->outputs());
        if (in_name == "XShape") {
          continue;
        }
        if (dims.size() >= 5) {
          change_image2d_to_cpu = true;
          break;
        }
        if (input_shape_default_) {
          auto max_dim =
              *std::max_element(dims.data().begin(), dims.data().end());
          if ((max_dim > image2d_max_width_size_) ||
              (dims.size() == 4 &&
               dims[1] * dims[3] / 4 > image2d_max_width_size_)) {
            change_image2d_to_cpu = true;
            break;
          }
        }
      }
      for (auto& var_name : op_info->input_names()) {
        auto* var = scope->FindVar(var_name);
        CHECK(var) << "no variable called " << var_name << " found";
        const auto dims = var->Get<Tensor>().dims();
        if (dims.size() >= 5) {
          change_image2d_to_cpu = true;
          break;
        }
        if (input_shape_default_) {
          auto max_dim =
              *std::max_element(dims.data().begin(), dims.data().end());
          if ((max_dim > image2d_max_width_size_) ||
              (dims.size() == 4 &&
               dims[1] * dims[3] / 4 > image2d_max_width_size_)) {
            change_image2d_to_cpu = true;
            break;
          }
        }
      }
      // image2d unsupport
      const std::vector<std::string> op_cases_unsupport_in_out_int{
          "elementwise_floordiv",
          "elementwise_add",
          "elementwise_sub",
          "elementwise_mul",
          "elementwise_div",
          "elementwise_pow",
          "elementwise_mod",
          "scale"};
      if (std::find(op_cases_unsupport_in_out_int.begin(),
                    op_cases_unsupport_in_out_int.end(),
                    op_type) != op_cases_unsupport_in_out_int.end()) {
        for (std::list<Node*>::iterator i = x->inlinks.begin();
             i != x->inlinks.end();
             ++i) {
          if ((*i)->arg()->type) {
            if ((*i)->arg()->type->precision() != PRECISION(kFP16) &&
                (*i)->arg()->type->precision() != PRECISION(kFloat)) {
              change_image2d_to_cpu = true;
            }
          }
        }
        for (std::list<Node*>::iterator i = x->outlinks.begin();
             i != x->outlinks.end();
             ++i) {
          if ((*i)->arg()->type) {
            if ((*i)->arg()->type->precision() != PRECISION(kFP16) &&
                (*i)->arg()->type->precision() != PRECISION(kFloat)) {
              change_image2d_to_cpu = true;
            }
          }
        }
      }

      // 4. if reduce op keepdim=false change target
      const std::vector<std::string> op_type_cases{"arg_max",
                                                   "reduce_max",
                                                   "reduce_min",
                                                   "reduce_mean",
                                                   "reduce_sum",
                                                   "reduce_prob",
                                                   "reduce_all",
                                                   "reduce_any"};
      if (std::find(op_type_cases.begin(), op_type_cases.end(), op_type) !=
          op_type_cases.end()) {
        auto op_teller = [&](const Node* node) -> bool {
          const std::vector<std::string> attr_names{"keep_dim", "keepdims"};
          auto* op_desc = const_cast<Node*>(node)->AsStmt().op_info();
          for (auto attr_name : attr_names) {
            if (op_desc->HasAttr(attr_name)) {
              if (op_desc->GetAttr<bool>(attr_name)) {
                return false;
              }
            }
          }
          return true;
        };
        change_image2d_to_cpu = change_image2d_to_cpu || op_teller(x);
      }

      // 5. split outlinks != 2 change target
      if (op_type == "split" && x->outlinks.size() != 2)
        change_image2d_to_cpu = true;

      // 6. gather X.dims.size() == 2
      if (op_type == "gather") {
        for (std::list<Node*>::iterator i = x->inlinks.begin();
             i != x->inlinks.end();
             ++i) {
          std::string in_name =
              get_argname((*i)->AsArg().name, inst.op_info()->inputs());
          if (in_name == "X") {
            auto* var = scope->FindVar((*i)->arg()->name);
            const auto& tensor = var->Get<Tensor>();
            if (tensor.dims().size() != 2) change_image2d_to_cpu = true;
          }
        }
      }

      // 7. reshape transpose change target
      if ((op_type == "reshape" || op_type == "reshape2") &&
          input_shape_default_) {
        change_image2d_to_buffer = true;
      }

      bool transpose_buffer =
          false;  // TODO(@sprouteer) transpose buffer poor performance
      if ((op_type == "transpose" || op_type == "transpose2") &&
          transpose_buffer) {
        for (std::list<Node*>::iterator i = x->inlinks.begin();
             i != x->inlinks.end();
             ++i) {
          std::string in_name =
              get_argname((*i)->AsArg().name, inst.op_info()->inputs());
          if (in_name == "X" && (*i)->inlinks.front()->IsStmt() &&
              (*i)->inlinks.front()->AsStmt().op_type() == "reshape2") {
            change_image2d_to_buffer = true;
          }
        }
        for (std::list<Node*>::iterator i = x->outlinks.begin();
             i != x->outlinks.end();
             ++i) {
          std::string out_name =
              get_argname((*i)->AsArg().name, inst.op_info()->outputs());
          if (out_name == "Out" && (*i)->outlinks.front()->IsStmt() &&
              (*i)->outlinks.front()->AsStmt().op_type() == "reshape2") {
            change_image2d_to_buffer = true;
          }
        }
      }
    }

    if (change_image2d_to_cpu) {
      UpdateTargetToCPU(x, graph->valid_places());
    } else if (change_image2d_to_buffer) {
      UpdateLayoutToBuffer(x);
    }
  }
}

void OpenCLMemoryObjectConfigPass::UpdateLayoutToBuffer(Node* x) {
  auto& inst = x->AsStmt();
  auto new_place = inst.place();

  new_place.target = TARGET(kOpenCL);
  new_place.precision = PRECISION(kFP16);
  new_place.layout = DATALAYOUT(kNCHW);

  std::vector<Place> places;
  places.push_back(new_place);
  inst.ResetKernels(places);
}

bool OpenCLMemoryObjectConfigPass::PrecTypeCompatible(const PrecisionType& p1,
                                                      const PrecisionType& p2) {
  if (p1 == p2 || p2 == PRECISION(kAny)) {
    return true;
  } else if ((p1 == PRECISION(kFP16) || p1 == PRECISION(kFloat)) &&
             (p2 == PRECISION(kFP16) || p2 == PRECISION(kFloat))) {
    return true;
  } else {
    return false;
  }
}

void OpenCLMemoryObjectConfigPass::UpdateTargetToCPU(
    Node* x, const std::vector<Place>& valid_places) {
  std::vector<std::string> arm_host_ops{"gather",
                                        "flatten2",
                                        "reshape2",
                                        "flatten",
                                        "reshape",
                                        "unsqueeze2",
                                        "unsqueeze",
                                        "squeeze2",
                                        "squeeze",
                                        "split"};
  std::vector<std::string> x86_host_ops{
      "abs",      "arg_max",         "cos",
      "exp",      "expand",          "flatten",
      "flatten2", "greater_than",    "hard_sigmoid",
      "log",      "pad2d",           "pixel_shuffle",
      "prelu",    "reshape",         "reshape2",
      "shape",    "shuffle_channel", "sin",
      "split",    "squeeze",         "squeeze2",
      "swish",    "unsqueeze",       "unsqueeze2",
      "yolo_box"};
  auto& inst = x->AsStmt();
  auto new_place = inst.place();
  new_place.layout = DATALAYOUT(kNCHW);
  const auto& op_type = inst.op_type();

  new_place.precision = PRECISION(kFloat);
  // target
  TargetType new_target = TARGET(kARM);
  for (const auto& place : valid_places) {
    if (place.target == TARGET(kARM)) {
      new_target = place.target;
      break;
    } else if (place.target == TARGET(kX86)) {
      new_target = place.target;
      break;
    }
  }

  if (new_target == TARGET(kARM)) {
    if (std::find(arm_host_ops.begin(), arm_host_ops.end(), op_type) !=
        arm_host_ops.end()) {
      new_target = TARGET(kHost);
    }
  } else if (new_target == TARGET(kX86)) {
    if (std::find(x86_host_ops.begin(), x86_host_ops.end(), op_type) !=
        x86_host_ops.end()) {
      new_target = TARGET(kHost);
    }
  }
  new_place.target = new_target;
  std::vector<Place> places;
  places.push_back(new_place);
  inst.ResetKernels(places);
  // match input type
  std::map<std::string, PrecisionType> in_types;
  for (std::list<Node*>::iterator i = x->inlinks.begin(); i != x->inlinks.end();
       ++i) {
    if ((*i)->arg()->type)
      in_types[(*i)->arg()->name] = (*i)->arg()->type->precision();
  }
  std::vector<std::string> in_names = inst.op_info()->input_names();
  std::vector<std::pair<float, std::unique_ptr<KernelBase>>> scored;
  bool type_match = false;
  for (auto&& kernel : inst.kernels()) {
    type_match = true;
    VLOG(2) << "opencl current candidate kernel is: " << kernel->summary();
    float score = 0;
    for (size_t i = 0; i < in_names.size(); ++i) {
      std::string tmp;
      CHECK(inst.op_info()->GetInputArgname(in_names[i], &tmp));
      if (!PrecTypeCompatible(in_types.at(in_names[i]),
                              kernel->GetInputDeclType(tmp)->precision())) {
        type_match = false;
      } else {
        score++;
      }
    }
    scored.emplace_back(score, std::move(kernel));
    if (type_match) break;
  }
  if (!type_match) LOG(WARNING) << "No kernel match all input precision";
  std::stable_sort(scored.begin(), scored.end(), KernelScoreCmp);
  inst.kernels().clear();
  inst.kernels().emplace_back(std::move(scored.front().second));
  for (auto&& kernel : inst.kernels()) {
    VLOG(2) << "opencl last candidate kernel is: " << kernel->summary();
  }
}

void OpenCLMemoryObjectConfigPass::SeparateOclMemoryObject(
    std::string* opencl_op_config,
    std::string* cpu_op_config,
    const std::string& ocl_memory_object_configs) {
  std::string buffer_flag = "device:gpu buffer";
  std::string cpu_flag = "device:cpu";
  int buffer_flag_size = buffer_flag.size();
  int cpu_flag_size = cpu_flag.size();
  auto gpu_op_begin = ocl_memory_object_configs.find("device:gpu buffer");
  auto cpu_op_begin = ocl_memory_object_configs.find("device:cpu");
  auto config_file_end = ocl_memory_object_configs.rfind("\n");

  if (gpu_op_begin != std::string::npos && cpu_op_begin == std::string::npos) {
    *opencl_op_config = ocl_memory_object_configs.substr(
        gpu_op_begin + buffer_flag_size, config_file_end - buffer_flag_size);
  } else if (gpu_op_begin == std::string::npos &&
             cpu_op_begin != std::string::npos) {
    *cpu_op_config = ocl_memory_object_configs.substr(
        cpu_op_begin + cpu_flag_size, config_file_end - cpu_flag_size);
  } else if (gpu_op_begin != std::string::npos &&
             cpu_op_begin != std::string::npos) {
    if (cpu_op_begin > gpu_op_begin) {
      *opencl_op_config = ocl_memory_object_configs.substr(
          gpu_op_begin + buffer_flag_size, cpu_op_begin - buffer_flag_size - 1);
      *cpu_op_config = ocl_memory_object_configs.substr(
          cpu_op_begin + cpu_flag_size,
          config_file_end - cpu_op_begin - cpu_flag_size);
    } else {
      *cpu_op_config = ocl_memory_object_configs.substr(
          cpu_op_begin + cpu_flag_size, gpu_op_begin - cpu_flag_size - 1);
      *opencl_op_config = ocl_memory_object_configs.substr(
          gpu_op_begin + buffer_flag_size,
          config_file_end - gpu_op_begin - buffer_flag_size);
    }
  }
}

void OpenCLMemoryObjectConfigPass::MemoryObjectConfig(SSAGraph* graph) {
  auto ocl_memory_object_configs = ReadMemoryObjectConfigsFromEnv();
  if (ocl_memory_object_configs.size() < 1) {
    return;
  }

  if (ocl_memory_object_configs.find("input shape:default") !=
      std::string::npos) {
    input_shape_default_ = true;
    if (ocl_memory_object_configs.find("gpu:Adreno"))
      image2d_max_width_size_ = 16384;
  }

  std::string opencl_op_config = "";
  std::string cpu_op_config = "";
  SeparateOclMemoryObject(
      &opencl_op_config, &cpu_op_config, ocl_memory_object_configs);

  auto buffer_nodes = GetNodesFromOpenCLOpConfig(graph, opencl_op_config);
  auto cpu_nodes = GetNodesFromOpenCLOpConfig(graph, cpu_op_config);

  VLOG(4) << "opencl buffer nodes size:" << buffer_nodes.size();
  VLOG(4) << "opencl cpu nodes size:" << cpu_nodes.size();
  for (auto& x : graph->mutable_nodes()) {
    auto in = x.inlinks.front();
    if (!in) {
      continue;
    }
    if (cpu_nodes.count(&x) != 0) {
      UpdateTargetToCPU(&x, graph->valid_places());
    } else if (buffer_nodes.count(&x) != 0) {
      UpdateLayoutToBuffer(&x);
    }
  }
}

}  // namespace mir
}  // namespace lite
}  // namespace paddle

REGISTER_MIR_PASS(opencl_memory_object_config_pass,
                  paddle::lite::mir::OpenCLMemoryObjectConfigPass)
    .BindTargets({TARGET(kOpenCL)});
