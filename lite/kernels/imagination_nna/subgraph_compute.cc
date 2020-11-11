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

#include "lite/kernels/imagination_nna/subgraph_compute.h"
#include <sys/time.h>
#include <time.h>
#include <limits>
#include <utility>
#include "lite/core/op_registry.h"
#include "lite/kernels/imagination_nna/bridges/graph.h"
#include "lite/kernels/imagination_nna/bridges/paddle_use_bridges.h"
#include "lite/kernels/imagination_nna/bridges/utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace imagination_nna {

bool SubgraphEngine::BuildDeviceProgram() {
  device_program_ready = false;
  int status = 0;
  // Convert all of ops and their input vars and weights and added into the NNA
  // IMG IR graph
  subgraph::imagination_nna::Graph graph{&imgdnn_mgr_};
  const auto& bridges = subgraph::SubgraphBridgeRegistry::Instance();
  if (!origin_program_) {
    BuildOriginProgram();
  }
  const auto& insts = origin_program_->instructions(kRootBlockIdx);
  for (auto& inst : insts) {
    auto op = const_cast<OpLite*>(inst.op());
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!bridges.Exists(op_type, TARGET(kImaginationNNA))) {
      return false;
    }
    auto kernel = inst.kernel();
    status |= bridges.Select(op_type, TARGET(kImaginationNNA))(
        reinterpret_cast<void*>(&graph),
        const_cast<OpLite*>(op),
        const_cast<KernelBase*>(kernel));
    if (subgraph::CHECK_FAILED(status)) {
      return false;
    }
  }

  // Collect the valid input and output nodes in the IMGDNN IR graph and update
  // the input and output names
  device_inames_.clear();
  std::vector<imgdnn_tensor> device_inodes;
  for (auto& input_name : input_names_) {
    if (graph.Has(input_name)) {
      device_inodes.push_back(graph.Get(input_name)->data());
      device_inames_.push_back(input_name);
    } else {
      LOG(WARNING) << "[NNA] Input node " << input_name
                   << " is ignored because it does not exist.";
    }
  }

  device_onames_.clear();
  std::vector<imgdnn_tensor> device_onodes;
  for (auto& output_name : output_names_) {
    if (graph.Has(output_name)) {
      device_onodes.push_back(graph.Get(output_name)->data());
      device_onames_.push_back(output_name);
    } else {
      LOG(WARNING) << "[NNA] Output node " << output_name
                   << " is ignored because it does not exist.";
    }
  }
  CHECK(!device_inames_.empty())
      << "[NNA] No input nodes found for building NNA model";
  CHECK(!device_onames_.empty())
      << "[NNA] No output nodes found for building NNA model";

  imgdnn_mgr_.CreateNetworkObject(device_inodes.size(),
                                  device_inodes.data(),
                                  device_onodes.size(),
                                  device_onodes.data());

  // inputs
  unsigned int num_inputs, num_outputs;
  imgdnn_mgr_.GetNetworkObjectInputs(
      std::numeric_limits<unsigned int>::max(), nullptr, &num_inputs);
  CHECK_EQ(num_inputs, device_inames_.size());
  device_itensors_.resize(num_inputs);
  imgdnn_mgr_.GetNetworkObjectInputs(
      num_inputs, device_itensors_.data(), nullptr);

  // show input info
  for (int i = 0; i < num_inputs; i++) {
    auto node = graph.Get(device_inames_[i]);
    auto type = node->type();
    auto layout = node->layout();
    VLOG(3) << "[NNA] Inputs[" << i << "] name: " << device_inames_[i]
            << " type: " << static_cast<int>(type)
            << " layout: " << DataLayoutToStr(layout);
  }

  // outputs
  imgdnn_mgr_.GetNetworkObjectOutputs(
      std::numeric_limits<unsigned int>::max(), nullptr, &num_outputs);
  CHECK_EQ(num_outputs, device_onames_.size());
  device_otensors_.resize(num_outputs);
  imgdnn_mgr_.GetNetworkObjectOutputs(
      num_outputs, device_otensors_.data(), nullptr);
  // show output info
  for (int i = 0; i < num_outputs; i++) {
    auto node = graph.Get(device_onames_[i]);
    auto type = node->type();
    auto layout = node->layout();
    VLOG(3) << "[NNA] Outputs[" << i << "] name: " << device_onames_[i]
            << " type: " << static_cast<int>(type)
            << " layout: " << DataLayoutToStr(layout);
    // Prepare the device output tensors
    switch (type) {
      case IMGDNN_TYPE_F32:
        origin_otensors_[i]->mutable_data<float>();
        break;
      case IMGDNN_TYPE_Q_I8:
      case IMGDNN_TYPE_Q_U8:
        origin_otensors_[i]->mutable_data<int8_t>();
        break;
      case IMGDNN_TYPE_I16:
        origin_otensors_[i]->mutable_data<int16_t>();
        break;
      case IMGDNN_TYPE_I32:
        origin_otensors_[i]->mutable_data<int32_t>();
        break;
      default:
        LOG(FATAL) << "[NNA] " << device_onames_[i]
                   << " can't mutable data with precision type "
                   << static_cast<int>(type);
        break;
    }
  }
  device_program_ready = true;

  return true;
}

bool SubgraphEngine::LaunchDeviceProgram() {
  if (!device_program_ready) {
    LOG(WARNING) << "[NNA] Build device program fail, run origin program";
    LaunchOriginProgram();
  }

  // Set input buffer
  for (size_t i = 0; i < origin_itensors_.size(); i++) {
    // check input shapes
    imgdnn_tensor_descriptor in_desc =
        imgdnn_mgr_.GetInputDescriptor(device_itensors_[i]);
    size_t in_size = imgdnn_mgr_.GetDescriptorSize(&in_desc);
    CHECK_EQ(in_size, origin_itensors_[i]->memory_size());

    auto origin_data = origin_itensors_[i]->mutable_data<int8_t>();
    auto converted_data = reinterpret_cast<uint8_t*>(origin_data);
    for (int j = 0; j < origin_itensors_[i]->data_size(); j++) {
      converted_data[j] =
          static_cast<uint8_t>(static_cast<int16_t>(origin_data[j]) + 128);
    }

    imgdnn_memory in_mem = imgdnn_mgr_.ImportMemory(
        static_cast<void*>(converted_data), origin_itensors_[i]->memory_size());
    imgdnn_mgr_.AddBindingInput(device_itensors_[i], in_mem);
  }

  // Set output buffer
  std::vector<imgdnn_memory> out_mems;
  for (size_t i = 0; i < origin_otensors_.size(); i++) {
    // check output shapes
    imgdnn_tensor_descriptor out_desc =
        imgdnn_mgr_.GetOutputDescriptor(device_otensors_[i]);
    size_t out_size = imgdnn_mgr_.GetDescriptorSize(&out_desc);
    CHECK_EQ(out_size, origin_otensors_[i]->memory_size());

    imgdnn_memory out_mem =
        imgdnn_mgr_.AllocateMemory(origin_otensors_[i]->memory_size());
    imgdnn_mgr_.AddBindingOutput(device_otensors_[i], out_mem);
    out_mems.push_back(out_mem);
  }

  // Run the img model by name
  imgdnn_mgr_.ExecuteNetworkObject(true, 0, nullptr, nullptr);

  // Copy the data of output tensor to the buffer of origin output tensors
  for (size_t i = 0; i < out_mems.size(); i++) {
    uint8_t* data = static_cast<uint8_t*>(
        imgdnn_mgr_.LockMemory(out_mems[i], IMGDNN_LOCK_ACCESS_READ_ONLY));

    int8_t* output_data = origin_otensors_[i]->mutable_data<int8_t>();
    for (size_t j = 0; j < origin_otensors_[i]->data_size(); j++) {
      output_data[j] = data[j] - 128;
    }
    imgdnn_mgr_.UnlockMemory(out_mems[i]);
    imgdnn_mgr_.DestroyMemory(out_mems[i]);
  }

  return true;
}

void SubgraphCompute::PrepareForRun() {
  auto& param = this->Param<param_t>();
  engine_.reset(new SubgraphEngine(ctx_.get(),
                                   param.block_idx,
                                   param.program_desc,
                                   param.exec_scope,
                                   param.input_data_names,
                                   param.output_data_names));
  CHECK(engine_);
}

void SubgraphCompute::Run() {
  CHECK(engine_);
  engine_->Run();
}

}  // namespace imagination_nna
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(subgraph,
                     kImaginationNNA,
                     kInt8,
                     kNCHW,
                     paddle::lite::kernels::imagination_nna::SubgraphCompute,
                     def)
    .BindInput("Inputs",
               {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .BindOutput("Outputs",
                {LiteType::GetTensorTy(TARGET(kHost), PRECISION(kInt8))})
    .Finalize();
