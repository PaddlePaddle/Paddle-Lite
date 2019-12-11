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

#include "lite/kernels/xpu/subgraph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <utility>
#include "lite/backends/xpu/device.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/xpu/bridges/graph.h"
#include "lite/kernels/xpu/bridges/paddle_use_bridges.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

int SubgraphEngine::BuildDeviceProgram() {
  int status = 0;
  // Convert all of input data vars and added into the XPU IR graph
  subgraph::xpu::Graph graph;
  for (auto& input_name : input_names_) {
    auto input_tensor = scope_->FindMutableTensor(input_name);
    CHECK(input_tensor);
    auto input_node =
        graph.AddNode(input_name, input_tensor->dims().Vectorize());
    CHECK(input_node);
    // XTCL doesn't support dynamic dimensions/shapes, so need to rebuild
    // the program when the shape of any input tensor is changed.
    status |= subgraph::REBUILD_WHEN_SHAPE_CHANGED;
  }
  // Convert all of ops and its weights and added into the XPU IR graph
  const auto& bridges = subgraph::Registry::Instance();
  for (auto& inst : origin_program_) {
    auto op = inst.op();
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists("XPU", op_type)) {
      return subgraph::FAILED;
    }
    status |= bridges.Select("XPU", op_type)(reinterpret_cast<void*>(&graph),
                                             const_cast<OpLite*>(op));
    if (subgraph::CHECK_FAILED(status)) {
      return subgraph::FAILED;
    }
  }
  // Obtain the output nodes of the XPU IR graph and build the graph to XPU
  // runtime
  std::vector<xtcl::xExpr*> output_nodes;
  for (auto& output_name : output_names_) {
    output_nodes.push_back(graph.GetNode(output_name).get());
  }
  device_program_ = lite::xpu::Device::Global().Build(
      &graph.builder_, &graph.params_, &output_nodes);
  if (device_program_ == nullptr) {
    LOG(WARNING) << "[XPU] Build model failed!";
    return subgraph::FAILED;
  }

  // Query and check the dimensions of input and output tensors
  origin_idims_.resize(input_names_.size());
  origin_itensors_.resize(input_names_.size());
  origin_odims_.resize(output_names_.size());
  origin_otensors_.resize(output_names_.size());
  for (int i = 0; i < input_names_.size(); i++) {
    origin_itensors_[i] = scope_->FindMutableTensor(input_names_[i]);
    CHECK(origin_itensors_[i]);
    origin_idims_[i] = origin_itensors_[i]->dims();
    VLOG(3) << "[XPU] Input dims[" << i << "]: " << origin_idims_[i];
  }
  for (int i = 0; i < output_names_.size(); i++) {
    origin_otensors_[i] = scope_->FindMutableTensor(output_names_[i]);
    CHECK(origin_otensors_[i]);
    origin_odims_[i] = origin_otensors_[i]->dims();
    VLOG(3) << "[XPU] Output dims[" << i << "]: " << origin_odims_[i];
  }
  return status;
}

int SubgraphEngine::LaunchDeviceProgram() {
  // Copy the data of origin input tensors to the buffer of input XPU tensors
  for (size_t i = 0; i < input_names_.size(); i++) {
    auto input_ndarray =
        xtcl::xNDArray::Empty(origin_itensors_[i]->dims().Vectorize(),
                              {kDLFloat, 32, 1},
                              {kDLCPU, 0});
    std::memcpy(static_cast<float*>(input_ndarray.ToDLPack()->dl_tensor.data),
                origin_itensors_[i]->mutable_data<float>(),
                sizeof(float) * origin_itensors_[i]->dims().production());
    device_program_->SetInputZeroCopy(input_names_[i],
                                      &input_ndarray.ToDLPack()->dl_tensor);
  }
  // Run the XPU model
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  auto start_time = GetCurrentUS();
  device_program_->Run();
  VLOG(3) << "[XPU] Process cost " << GetCurrentUS() - start_time << " us";
  // Copy the data of output XPU tensor to the buffer of origin output tensors
  for (size_t i = 0; i < output_names_.size(); i++) {
    auto output_ndarray = device_program_->GetOutput(i);
    std::memcpy(origin_otensors_[i]->mutable_data<float>(),
                static_cast<float*>(output_ndarray.ToDLPack()->dl_tensor.data),
                sizeof(float) * origin_otensors_[i]->dims().production());
  }
  return 0;
}

void SubgraphCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  engine_.reset(new SubgraphEngine(param.sub_block_idx,
                                   &param.sub_block_desc,
                                   param.input_data_names,
                                   param.output_data_names,
                                   param.scope));
  CHECK(engine_);
  engine_->Build();
}

void SubgraphCompute::Run() {
  CHECK(engine_);
  engine_->Launch();
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kXPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::xpu::SubgraphCompute,
                     def)
    .BindInput("Inputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Outputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
