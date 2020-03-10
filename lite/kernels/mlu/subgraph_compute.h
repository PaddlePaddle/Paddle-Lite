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
                 ::paddle::lite_api::PrecisionType type)
      : subgraph::Engine(
            ctx, block_idx, block_desc, input_names, output_names, scope) {
    graph_.SetFPType(type);
  }

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

 protected:
  int BuildDeviceProgram() override {
    int status = 0;
    // Convert all of input data vars and added into the MLU IR graph
    for (auto& input_name : input_names_) {
      auto input_tensor = scope_->FindMutableTensor(input_name);
      CHECK(input_tensor);
      auto input_node =
          graph_.AddNode(input_name,
                         input_tensor->dims().Vectorize(),
                         CNML_TENSOR,
                         CNML_NHWC,
                         graph_.FPType(),
                         const_cast<void*>(input_tensor->raw_data()));
      CHECK(input_node);
      // MLU doesn't support dynamic dimensions/shapes, so need to rebuild
      // the program when the shape of any input tensor is changed.
      status |= subgraph::REBUILD_WHEN_SHAPE_CHANGED;
    }
    LOG(INFO) << "START TO CONVERT ";
    // Convert all of ops and its weights and added into the MLU IR graph
    const auto& bridges = subgraph::Registry::Instance();
    for (auto& inst : origin_program_) {
      auto op = inst.op();
      CHECK(op);
      op->CheckShape();
      op->InferShape();
      std::string op_type = op->op_info()->Type();
      if (!bridges.Exists(op_type, TARGET(kMLU))) {
        LOG(INFO) << "MLU bridges doesn't support op_type: " << op_type;
        return subgraph::FAILED;
      }
      auto kernel = inst.kernel();
      status |= bridges.Select(op_type, TARGET(kMLU))(
          reinterpret_cast<void*>(&graph_),
          const_cast<OpLite*>(op),
          const_cast<KernelBase*>(kernel));
      if (subgraph::CHECK_FAILED(status)) {
        return subgraph::FAILED;
      }
    }
    // Obtain the output nodes of the MLU IR graph and build the graph to MLU
    // runtime
    std::vector<std::string> valid_output_names;
    for (auto& output_name : output_names_) {
      if (graph_.HasNode(output_name)) {
        graph_.AddOutput(graph_.GetNode(output_name));
        auto output_tensor = scope_->FindMutableTensor(output_name);
        void* p_data = static_cast<void*>(
            output_tensor->mutable_data<typename ::paddle::lite::subgraph::mlu::
                                            FPTypeTraits<Precision>::T>(
                TARGET(kMLU)));
        auto node = graph_.GetNode(output_name);
        CHECK(p_data);
        node->set_mlu_ptr(p_data);
        valid_output_names.push_back(output_name);
      }
    }
    for (auto& input_name : input_names_) {
      graph_.AddInput(graph_.GetNode(input_name));
    }
    CHECK(!valid_output_names.empty()) << "[MLU] no valid output names";
    auto& mlu_context = this->ctx_->template As<MLUContext>();
    auto core_version = mlu_context.MLUCoreVersion();
    auto core_number = mlu_context.MLUCoreNumber();
    graph_.Compile(core_version, core_number);
    return status;
  }

  int LaunchDeviceProgram() override {
    auto& mlu_context = this->ctx_->template As<MLUContext>();
    auto exec_queue = mlu_context.exec_queue();
    u32_t affinity = mlu_context.affinity();
    cnrtInvokeFuncParam_t forward_param = mlu_context.forward_param();
    int data_param = 1;
    forward_param.data_parallelism = &data_param;
    forward_param.affinity = &affinity;
    forward_param.end = CNRT_PARAM_END;
    graph_.Compute(forward_param, exec_queue);
    return 0;
  }

  paddle::lite::subgraph::mlu::Graph graph_;
};

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
