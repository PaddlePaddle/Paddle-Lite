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

#include "lite/kernels/npu/subgraph_engine.h"
#include <sys/time.h>
#include <time.h>
#include <utility>
#include "lite/backends/npu/device.h"
#include "lite/core/mir/subgraph/subgraph_bridge_registry.h"
#include "lite/kernels/npu/bridges/graph.h"
#include "lite/kernels/npu/bridges/paddle_use_bridges.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace npu {

int Engine::BuildDeviceProgram() {
  int status = 0;
  // Convert all of input data vars and added into the HiAI IR graph
  Graph graph;
  for (auto& input_name : input_names_) {
    auto input_tensor = scope_->FindMutableTensor(input_name);
    CHECK(input_tensor);
    auto input_node =
        graph.AddNode(input_name, input_tensor->dims().Vectorize());
    CHECK(input_node);
    // HiAI DDK doesn't support dynamic dimensions/shapes, so need to rebuild
    // the program when the shape of any input tensor is changed.
    status |= subgraph::REBUILD_WHEN_SHAPE_CHANGED;
  }
  // Convert all of ops and its weights and added into the HiAI IR graph
  const auto& bridges = subgraph::Registry::Instance();
  for (auto& inst : origin_program_) {
    auto op = inst.op();
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists("NPU", op_type)) {
      return subgraph::FAILED;
    }
    status |= bridges.Select("NPU", op_type)(reinterpret_cast<void*>(&graph),
                                             const_cast<OpLite*>(op));
    if (subgraph::CHECK_FAILED(status)) {
      return subgraph::FAILED;
    }
  }
  // Set the input and output nodes of the HiAI IR graph
  std::vector<ge::Operator> input_nodes, output_nodes;
  for (auto& input_name : input_names_) {
    input_nodes.push_back(*graph.GetNode(input_name));
  }
  for (auto& output_name : output_names_) {
    output_nodes.push_back(*graph.GetNode(output_name));
  }
  // Build the HiAI IR graph to HiAI om model
  device_program_ =
      lite::npu::Device::Global().Build(model_name_, input_nodes, output_nodes);
  if (device_program_ == nullptr) {
    LOG(WARNING) << "[NPU] Build model failed!";
    return subgraph::FAILED;
  }

  // Query and check the dimensions of input and output tensors
  std::vector<hiai::TensorDimension> device_idims, device_odims;
  if (device_program_->GetModelIOTensorDim(
          model_name_, device_idims, device_odims) != hiai::AI_SUCCESS) {
    LOG(WARNING)
        << "[NPU] Get the dimensions of input and output tensors failed!";
    return subgraph::FAILED;
  }
  CHECK_EQ(device_idims.size(), input_names_.size());
  CHECK_EQ(device_odims.size(), output_names_.size());
  origin_idims_.resize(input_names_.size());
  origin_itensors_.resize(input_names_.size());
  device_idatasizes_.resize(input_names_.size());
  device_itensors_.resize(input_names_.size());
  origin_odims_.resize(output_names_.size());
  origin_otensors_.resize(output_names_.size());
  device_odatasizes_.resize(output_names_.size());
  device_otensors_.resize(output_names_.size());
  for (int i = 0; i < input_names_.size(); i++) {
    origin_itensors_[i] = scope_->FindMutableTensor(input_names_[i]);
    CHECK(origin_itensors_[i]);
    origin_idims_[i] = origin_itensors_[i]->dims();
    VLOG(3) << "[NPU] Input dims[" << i << "]: {" << device_idims[i].GetNumber()
            << "," << device_idims[i].GetChannel() << ","
            << device_idims[i].GetHeight() << "," << device_idims[i].GetWidth()
            << "}";
    device_idatasizes_[i] =
        device_idims[i].GetNumber() * device_idims[i].GetChannel() *
        device_idims[i].GetHeight() * device_idims[i].GetWidth();
    CHECK_EQ(device_idatasizes_[i], origin_idims_[i].production());
    device_itensors_[i].reset(new hiai::AiTensor);
    device_itensors_[i]->Init(&(device_idims[i]));
  }
  for (int i = 0; i < output_names_.size(); i++) {
    origin_otensors_[i] = scope_->FindMutableTensor(output_names_[i]);
    CHECK(origin_otensors_[i]);
    origin_odims_[i] = origin_otensors_[i]->dims();
    VLOG(3) << "[NPU] Output dims[" << i << "]: {"
            << device_odims[i].GetNumber() << ","
            << device_odims[i].GetChannel() << ","
            << device_odims[i].GetHeight() << "," << device_odims[i].GetWidth()
            << "}";
    device_odatasizes_[i] =
        device_odims[i].GetNumber() * device_odims[i].GetChannel() *
        device_odims[i].GetHeight() * device_odims[i].GetWidth();
    CHECK_EQ(device_odatasizes_[i], origin_odims_[i].production());
    device_otensors_[i].reset(new hiai::AiTensor);
    device_otensors_[i]->Init(&(device_odims[i]));
  }
  return status;
}

int Engine::LaunchDeviceProgram() {
  // Copy the data of origin input tensors to the buffer of input HiAI tensors
  for (size_t i = 0; i < input_names_.size(); i++) {
    std::memcpy(static_cast<float*>(device_itensors_[i]->GetBuffer()),
                origin_itensors_[i]->mutable_data<float>(),
                sizeof(float) * static_cast<size_t>(device_idatasizes_[i]));
  }
  // Run the HiAI model by name
  std::string key = "model_name";  // Note: key seems must be model_name
  model_context_.AddPara(key, model_name_);
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  int istamp;
  auto start_time = GetCurrentUS();
  CHECK_EQ(
      device_program_->Process(
          model_context_, device_itensors_, device_otensors_, 1000, istamp),
      hiai::AI_SUCCESS);
  VLOG(3) << "[NPU] Process cost " << GetCurrentUS() - start_time << " us";
  // Copy the data of output HiAI tensor to the buffer of origin output tensors
  for (size_t i = 0; i < output_names_.size(); i++) {
    std::memcpy(origin_otensors_[i]->mutable_data<float>(),
                static_cast<float*>(device_otensors_[i]->GetBuffer()),
                sizeof(float) * static_cast<size_t>(device_odatasizes_[i]));
  }
  return 0;
}

}  // namespace npu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
