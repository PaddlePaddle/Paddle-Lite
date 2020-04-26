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

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "lite/api/paddle_place.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"
#include "lite/core/types.h"
#include "lite/kernels/mlu/bridges/graph.h"
#include "lite/kernels/npu/bridges/engine.h"
#include "lite/kernels/npu/bridges/registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace mlu {

template <PrecisionType Precision>
class SubgraphEngine : public subgraph::Engine {
 public:
  SubgraphEngine(KernelContext* ctx,
                 int block_idx,
                 cpp::BlockDesc* block_desc,
                 const std::vector<std::string>& input_names,
                 const std::vector<std::string>& output_names,
                 Scope* scope,
                 paddle::lite_api::PrecisionType type)
      : subgraph::Engine(
            ctx, block_idx, block_desc, input_names, output_names, scope),
        fp_type_(type) {}

  int Build() {
    // In order to attach all of the ops of the block desc, we need to build
    // the original program firstly.
    BuildOriginProgram();
    // Run InferShape() of all of ops, and convert Paddle ops to MLU IR graph
    build_device_program_status_ = BuildDeviceProgram();
    return build_device_program_status_;
  }

  int Launch() {
    // Rebuild device program when the shapes of input tensors have been
    // changed.
    if (subgraph::CHECK_SUCCESS(build_device_program_status_) &&
        subgraph::CHECK_REBUILD_WHEN_SHAPE_CHANGED(
            build_device_program_status_) &&
        InputShapeChanged()) {
      Build();
    }
    if (subgraph::CHECK_FAILED(build_device_program_status_)) {
      LaunchOriginProgram();
    } else {
      LaunchDeviceProgram();
    }
    return 0;
  }

  bool InputShapeChanged() {
    std::vector<std::vector<int64_t>> new_shape;
    for (auto origin_itensor : origin_itensors_) {
      new_shape.push_back(origin_itensor->dims().Vectorize());
    }
    inputs_shape_ = new_shape;
    if (shape_graph_map_.count(inputs_shape_) > 0) {
      return false;
    }
    return true;
  }

 protected:
  int BuildDeviceProgram() override {
    int status = 0;
    auto graph = std::make_shared<paddle::lite::subgraph::mlu::Graph>();
    graph->SetFPType(fp_type_);
    std::vector<std::vector<int64_t>> new_shape;
    origin_itensors_.clear();
    origin_otensors_.clear();

    // Convert all of input data vars and added into the MLU IR graph
    status |= subgraph::REBUILD_WHEN_SHAPE_CHANGED;
    for (auto& input_name : input_names_) {
      auto input_tensor = scope_->FindMutableTensor(input_name);

      origin_itensors_.push_back(input_tensor);
      new_shape.push_back(input_tensor->dims().Vectorize());

      CHECK(input_tensor);
      auto input_node = graph->AddNode(input_name,
                                       input_tensor->dims().Vectorize(),
                                       CNML_TENSOR,
                                       CNML_NCHW,
                                       graph->FPType());
      CHECK(input_node);
      // MLU doesn't support dynamic dimensions/shapes, so need to rebuild
      // the program when the shape of any input tensor is changed.
    }
    LOG(INFO) << "START TO CONVERT ";
    // Convert all of ops and its weights and added into the MLU IR graph
    const auto& bridges = subgraph::Registry::Instance();
    for (auto& inst : origin_program_) {
      auto op = inst.op();
      CHECK(op);
      std::string op_type = op->op_info()->Type();
      op->CheckShape();
      const_cast<OpLite*>(op)->InferShape();
      if (!bridges.Exists(op_type, TARGET(kMLU))) {
        LOG(INFO) << "MLU bridges doesn't support op_type: " << op_type;
        return subgraph::FAILED;
      }
      auto kernel = inst.kernel();
      status |= bridges.Select(op_type, TARGET(kMLU))(
          reinterpret_cast<void*>(graph.get()),
          const_cast<OpLite*>(op),
          const_cast<KernelBase*>(kernel));
      if (subgraph::CHECK_FAILED(status)) {
        return subgraph::FAILED;
      }
    }
    // Obtain the output nodes of the MLU IR graph and build the graph to MLU
    // runtime
    for (auto& output_name : output_names_) {
      if (graph->HasNode(output_name)) {
        graph->AddOutput(graph->GetNode(output_name));
        auto output_tensor = scope_->FindMutableTensor(output_name);
        origin_otensors_.push_back(output_tensor);

        // auto node = graph->GetNode(output_name);
        // CHECK(p_data);
        // node->set_mlu_ptr(p_data);
      }
    }
    for (auto& input_name : input_names_) {
      graph->AddInput(graph->GetNode(input_name));
    }

    CHECK(!origin_otensors_.empty()) << "[MLU] no valid output names";
    auto& mlu_context = this->ctx_->template As<MLUContext>();
    auto core_version = mlu_context.MLUCoreVersion();
    auto core_number = mlu_context.MLUCoreNumber();
    graph->Compile(core_version, core_number);
    shape_graph_map_[new_shape] = graph;
    if (GetBoolFromEnv("SAVE_MLU_OFFLINE_MODEL")) {
      graph->GenOfflineModel(GetOfflineModName());
    }
    return status;
  }

  std::string TrimStrings(std::string origin_str) {
    std::string str = origin_str;
    std::size_t found = str.find("0x");
    std::size_t found_end = 0;
    std::vector<std::string> del_strs = {
        "/trans_io_copy", "/trans_cast", "/trans_layout"};
    for (auto iterm : del_strs) {
      found_end = str.find(iterm);
      // trim point address and one of the del_strs
      if (found != std::string::npos && found_end != std::string::npos) {
        str.replace(found, found_end - found, "");
        found_end = str.find(iterm);
        str.replace(found_end, iterm.size(), "");
        break;
      }
    }
    return str;
  }

  std::string GetOfflineModName() {
    sort(input_names_.begin(), input_names_.end());
    sort(output_names_.begin(), output_names_.end());
    std::string name = "";
    std::string delimiter = "__";
    std::string delimiter_num = "_";
    std::string tmp = "";
    for (auto input_name : input_names_) {
      tmp = input_name;
      name += TrimStrings(tmp) + delimiter + "input_shape_";
      auto input_tensor = scope_->FindMutableTensor(input_name);
      for (auto iterm : input_tensor->dims().Vectorize()) {
        name += std::to_string(iterm) + delimiter_num;
      }
      name += delimiter;
    }
    for (auto output_name : output_names_) {
      tmp = output_name;
      name += TrimStrings(tmp) + delimiter + "output_shape_";
      auto output_tensor = scope_->FindMutableTensor(output_name);
      for (auto iterm : output_tensor->dims().Vectorize()) {
        name += std::to_string(iterm) + delimiter_num;
      }
      name += delimiter;
    }
    std::replace(name.begin(), name.end(), '/', '-');
    return name;
  }

  int LaunchDeviceProgram() override {
    // prepare input and output memory
    auto graph = shape_graph_map_[inputs_shape_];
    auto* graph_input = graph->MutableInputs();
    auto* graph_output = graph->MutableOutputs();
    CHECK_EQ(graph_input->size(), origin_itensors_.size());
    CHECK_EQ(graph_output->size(), origin_otensors_.size());

    for (size_t i = 0; i < origin_itensors_.size(); ++i) {
      graph_input->at(i)->set_mlu_ptr(
          const_cast<void*>(origin_itensors_[i]->raw_data()));
    }
    for (size_t i = 0; i < origin_otensors_.size(); ++i) {
      origin_otensors_[i]->Resize(graph_output->at(i)->get_origin_shape());
      void* p_data = static_cast<void*>(
          origin_otensors_[i]
              ->mutable_data<typename paddle::lite::subgraph::mlu::FPTypeTraits<
                  Precision>::T>(TARGET(kMLU)));
      graph_output->at(i)->set_mlu_ptr(p_data);
    }

    auto& mlu_context = this->ctx_->template As<MLUContext>();
    auto exec_queue = mlu_context.exec_queue();
    u32_t affinity = mlu_context.affinity();
    cnrtInvokeFuncParam_t forward_param = mlu_context.forward_param();
    int data_param = 1;
    forward_param.data_parallelism = &data_param;
    forward_param.affinity = &affinity;
    forward_param.end = CNRT_PARAM_END;

    graph->Compute(forward_param, exec_queue);

    // // =========== DUMP ===================
    // for (auto input_name : input_names_) {
    //   auto input_tensor =
    //   shape_graph_map_[inputs_shape_]->GetNode(input_name);
    //   auto dump_name = input_name;
    //   while (dump_name.find("/") != std::string::npos) {
    //     dump_name = dump_name.replace(dump_name.find("/"), 1, "_");
    //   }
    //   VLOG(6) << "dump_name: " << dump_name;
    //   input_tensor->ToFile(dump_name);
    // }
    // for (auto output_name : output_names_) {
    //   if (shape_graph_map_[inputs_shape_]->HasNode(output_name)) {
    //     auto output_tensor =
    //     shape_graph_map_[inputs_shape_]->GetNode(output_name);
    //     auto dump_name = output_name;
    //     while (dump_name.find("/") != std::string::npos) {
    //       dump_name = dump_name.replace(dump_name.find("/"), 1, "_");
    //     }
    //     VLOG(6) << "dump_name: " << dump_name;
    //     output_tensor->ToFile(dump_name);
    //   } else {
    //     VLOG(6) << "graph does not have " << output_name << " as output"
    //             << std::endl;
    //   }
    // }
    // // =========== DUMP END ================
    return 0;
  }

  paddle::lite_api::PrecisionType fp_type_;
  std::vector<std::vector<int64_t>> inputs_shape_{};
  std::map<std::vector<std::vector<int64_t>>,
           std::shared_ptr<paddle::lite::subgraph::mlu::Graph>>
      shape_graph_map_{};
};  // namespace mlu

template <PrecisionType Precision>
class SubgraphCompute
    : public KernelLite<TARGET(kMLU), Precision, DATALAYOUT(kNHWC)> {
 public:
  using param_t = operators::SubgraphParam;

  void PrepareForRun() override {
    auto& param = this->template Param<param_t>();
    // LOG(INFO) << "SUBGRAP Prepare RUN index " << param.sub_block_idx;
    engine_.reset(new SubgraphEngine<Precision>(this->ctx_.get(),
                                                param.sub_block_idx,
                                                param.sub_block_desc,
                                                param.input_data_names,
                                                param.output_data_names,
                                                param.scope,
                                                this->precision()));
    CHECK(engine_);
    engine_->Build();
  }

  void Run() override {
    CHECK(engine_);
    engine_->Launch();
  }

  virtual ~SubgraphCompute() = default;

 private:
  std::unique_ptr<SubgraphEngine<Precision>> engine_;
};

}  // namespace mlu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
