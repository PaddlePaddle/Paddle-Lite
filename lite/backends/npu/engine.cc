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

#include "lite/backends/npu/engine.h"
#include <mutex>  // NOLINT
#include <utility>
#include "lite/backends/npu/bridges/paddle_use_bridges.h"  // NOLINT
#include "lite/backends/npu/bridges/registry.h"            // NOLINT
#include "lite/backends/npu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace npu {

int Engine::CreateDeviceProgram() {
  int status = 0;
  // Convert all of valid input vars and added into the HiAI IR graph
  bridges::cvt_ctx_type ctx;
  const auto& bridges = bridges::Factory::Instance();
  const auto& supported_lists = bridges.AllFunctions();
  for (auto& input_name : input_names_) {
    auto input_tensor = scope_->FindMutableTensor(input_name);
    CHECK(input_tensor);
    ge::TensorDesc input_desc(ge::Shape(input_tensor->dims().Vectorize()),
                              ge::Format::FORMAT_NCHW,
                              ge::DataType::DT_FLOAT);
    auto input_node = ctx.AddNode<ge::op::Data>(input_name);
    CHECK(input_node);
    input_node->update_input_desc_x(input_desc);
    // HiAI DDK doesn't support dynamic dimensions/shapes, so need to rebuild
    // the program when the shape of any input tensor is changed.
    status |= bridges::REBUILD_WHEN_SHAPE_CHANGED;
  }
  // Convert all of ops and its weights and added into the HiAI IR graph
  for (auto& inst : origin_program_) {
    auto op = inst.op();
    CHECK(op);
    op->CheckShape();
    op->InferShape();
    std::string op_type = op->op_info()->Type();
    if (!supported_lists.count(op_type)) {
      return bridges::FAILED;
    }
    status |= supported_lists.at(op_type)(&ctx, const_cast<lite::OpLite*>(op));
    if (bridges::CHECK_FAILED(status)) {
      return bridges::FAILED;
    }
  }
  // Set the input and output nodes of the HiAI IR graph and build the graph to
  // HiAI om model
  std::vector<ge::Operator> input_nodes, output_nodes;
  for (auto& input_name : input_names_) {
    CHECK(ctx.HasNode(input_name));
    input_nodes.push_back(*ctx.GetNode(input_name));
  }
  for (auto& output_name : output_names_) {
    CHECK(ctx.HasNode(output_name));
    output_nodes.push_back(*ctx.GetNode(output_name));
  }
  ge::Graph ir_graph("graph");
  ir_graph.SetInputs(input_nodes).SetOutputs(output_nodes);
  ge::Model om_model("model", "model");
  om_model.SetGraph(ir_graph);
  domi::HiaiIrBuild ir_build;
  domi::ModelBufferData om_model_buf;
  if (!ir_build.CreateModelBuff(om_model, om_model_buf)) {
    LOG(WARNING) << "[NPU] CreateModelBuff failed!";
    return bridges::FAILED;
  }
  if (!ir_build.BuildIRModel(om_model, om_model_buf)) {
    LOG(WARNING) << "[NPU] BuildIRModel failed!";
    ir_build.ReleaseModelBuff(om_model_buf);
    return bridges::FAILED;
  }
  // Create a HiAI model manager client to load HiAI om model
  device_program_.reset(new hiai::AiModelMngerClient());
  if (device_program_->Init(nullptr) != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] AiModelMngerClient init failed)!";
    ir_build.ReleaseModelBuff(om_model_buf);
    return bridges::FAILED;
  }
  model_name_ = "model_" + std::to_string(block_idx_) + ".om";
  auto model_desc = std::make_shared<hiai::AiModelDescription>(
      model_name_,
      Device::Global().freq_level(),
      Device::Global().framework_type(),
      Device::Global().model_type(),
      Device::Global().device_type());
  model_desc->SetModelBuffer(om_model_buf.data, om_model_buf.length);
  std::vector<std::shared_ptr<hiai::AiModelDescription>> model_descs;
  model_descs.push_back(model_desc);
  if (device_program_->Load(model_descs) != hiai::AI_SUCCESS) {
    LOG(WARNING) << "[NPU] AiModelMngerClient load model failed!";
    ir_build.ReleaseModelBuff(om_model_buf);
    return bridges::FAILED;
  }
  ir_build.ReleaseModelBuff(om_model_buf);
  return status;
}

int Engine::BuildDeviceProgram() {
  int status = CreateDeviceProgram();
  if (bridges::CHECK_FAILED(status)) {
    return bridges::FAILED;
  }
  // Query and check the dimensions of input and output tensors
  std::vector<hiai::TensorDimension> device_idims, device_odims;
  if (device_program_->GetModelIOTensorDim(
          model_name_, device_idims, device_odims) != hiai::AI_SUCCESS) {
    LOG(WARNING)
        << "[NPU] Get the dimensions of input and output tensors failed!";
    return bridges::FAILED;
  }
  CHECK_EQ(device_idims.size(), input_names_.size());
  CHECK_EQ(device_odims.size(), output_names_.size());
  origin_idims_.resize(device_idims.size());
  origin_itensors_.resize(device_idims.size());
  device_idatasizes_.resize(device_idims.size());
  device_itensors_.resize(device_idims.size());
  origin_odims_.resize(device_odims.size());
  origin_otensors_.resize(device_odims.size());
  device_odatasizes_.resize(device_odims.size());
  device_otensors_.resize(device_odims.size());
  for (int i = 0; i < input_names_.size(); i++) {
    origin_itensors_[i] = scope_->FindMutableTensor(input_names_[i]);
    CHECK(origin_itensors_[i]);
    origin_idims_[i] = origin_itensors_[i]->dims();
    VLOG(3) << "[NPU] Origin input dims[" << i << "]: " << origin_idims_[i];
    VLOG(3) << "[NPU] Device input dims[" << i << "]: {"
            << device_idims[i].GetNumber() << ","
            << device_idims[i].GetChannel() << ","
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
    VLOG(3) << "[NPU] Origin output dims[" << i << "]: " << origin_odims_[i];
    VLOG(3) << "[NPU] Device output dims[" << i << "]: {"
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

int Engine::RunDeviceProgram() {
  // Copy the data of origin input tensors to the buffer of input HiAI tensors
  for (size_t i = 0; i < device_itensors_.size(); i++) {
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
  for (size_t i = 0; i < device_otensors_.size(); i++) {
    std::memcpy(origin_otensors_[i]->mutable_data<float>(),
                static_cast<float*>(device_otensors_[i]->GetBuffer()),
                sizeof(float) * static_cast<size_t>(device_odatasizes_[i]));
  }
  return 0;
}

int Engine::CreateOriginProgram() {
  // TODO(hong19860320) The block_desc need to be divided into subgraphs during
  // the exection time. But only see them as a subgraph now.
  origin_program_.clear();
  for (int op_idx = 0; op_idx < block_desc_->OpsSize(); op_idx++) {
    auto op_desc = block_desc_->GetOp<lite::cpp::OpDesc>(op_idx);
    CHECK(op_desc);
    std::string op_type = op_desc->Type();
    auto op = lite::LiteOpRegistry::Global().Create(op_desc->Type());
    op->Attach(*op_desc, scope_);
    std::unique_ptr<KernelBase> picked_kernel;
    if (op_desc->HasAttr(lite::kKernelTypeAttr)) {
      // Create op and pick up kernel according to the kKernelTypeAttr attribute
      auto kernel_type = op_desc->GetAttr<std::string>(lite::kKernelTypeAttr);
      std::string alias;
      Place place;
      lite::KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
      VLOG(3) << "[NPU] Found the attr '" << kKernelTypeAttr
              << "': " << kernel_type << " for " << op_type;
      auto kernels = op->CreateKernels({place});
      CHECK_GT(kernels.size(), 0) << "No kernels found for " << op_type;
      auto it = std::find_if(kernels.begin(),
                             kernels.end(),
                             [&](std::unique_ptr<lite::KernelBase>& it) {
                               return it->alias() == alias;
                             });
      CHECK(it != kernels.end());
      picked_kernel = std::move(*it);
    } else {
      VLOG(3) << "[NPU] The attr '" << kKernelTypeAttr
              << "' not found, pick the first kernel for " << op_type;
      auto kernels = op->CreateKernels({lite::Place{TARGET(kARM)}});
      CHECK_GT(kernels.size(), 0) << "No kernels found for " << op_type;
      picked_kernel = std::move(kernels.front());
    }
    picked_kernel->SetContext(
        ContextScheduler::Global().NewContext(picked_kernel->target()));
    origin_program_.emplace_back(std::move(op), std::move(picked_kernel));
  }
  return 0;
}

int Engine::BuildOriginProgram() { return CreateOriginProgram(); }

int Engine::RunOriginProgram() {
  for (auto& inst : origin_program_) {
    auto op_type = inst.op()->op_info()->Type();
    if (op_type == "feed" || op_type == "fetch") continue;
    inst.Run();
  }
  return 0;
}

int Engine::Build() {
  // Need build original program before to attach all of the ops of the block
  // desc
  /* build_target_program_status_ = */ BuildOriginProgram();
  // Run InferShape() of all of ops, and convert Paddle ops to HiAI om model,
  // then load HiAI om model as the device program
  build_device_program_status_ = BuildDeviceProgram();
}

bool Engine::InputShapeChanged() {
  for (int i = 0; i < origin_itensors_.size(); i++) {
    if (origin_itensors_[i]->dims() != origin_idims_[i]) {
      return true;
    }
  }
  return false;
}

int Engine::Run() {
  // Rebuild device program when the shapes of input tensors have been changed.
  if (bridges::CHECK_SUCCESS(build_device_program_status_) &&
      bridges::CHECK_REBUILD_WHEN_SHAPE_CHANGED(build_device_program_status_) &&
      InputShapeChanged()) {
    Build();
  }
  if (bridges::CHECK_FAILED(build_device_program_status_)) {
    RunOriginProgram();
  } else {
    RunDeviceProgram();
  }
  return 0;
}

}  // namespace npu
}  // namespace lite
}  // namespace paddle
