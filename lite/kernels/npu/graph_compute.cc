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

#include "lite/kernels/npu/graph_compute.h"
#include <sys/time.h>
#include <time.h>

namespace paddle {
namespace lite {
namespace kernels {
namespace npu {

void GraphCompute::PrepareForRun() {
  auto& ctx = this->ctx_->template As<NPUContext>();
  auto& param = this->Param<param_t>();

  // Load HiAI model from the weight tensor and release its buffer
  // to save memory
  CHECK(param.weight);
  CHECK(lite::npu::LoadModel(*param.weight, &model_client_, &model_name_));
  // TODO(hong19860320): find an good way to free the model data.
  // No interface exists to free the data of tensor, so I resize the dim to 1
  // and change target to force it to realloc a small size memory.
  param.weight->Resize({1});
  param.weight->mutable_data<int8_t>(TargetType::kARM);
  CHECK(model_client_);

  // Query the dimensions of NPU input and output tensors from HiAI model
  std::vector<hiai::TensorDimension> npu_idims;
  std::vector<hiai::TensorDimension> npu_odims;
  int ret =
      model_client_->GetModelIOTensorDim(model_name_, npu_idims, npu_odims);
  CHECK_EQ(ret, hiai::AI_SUCCESS)
      << "[NPU] Get the dimensions of input and output tensors failed.";

  // Check whether the data sizes of NPU input and output tensors are the
  // same as CPU's, then create and initialize NPU input and output tensors.
  npu_itensors_.resize(npu_idims.size());
  npu_otensors_.resize(npu_odims.size());
  npu_idatasizes_.resize(npu_idims.size());
  npu_odatasizes_.resize(npu_odims.size());
  for (size_t i = 0; i < npu_idims.size(); ++i) {
    auto cpu_itensor = param.inputs[i].second;
    CHECK(cpu_itensor);
    VLOG(3) << "[NPU] CPU input dims[" << i << "]: " << cpu_itensor->dims();
    VLOG(3) << "[NPU] NPU input dims[" << i << "]: {"
            << npu_idims[i].GetNumber() << "," << npu_idims[i].GetChannel()
            << "," << npu_idims[i].GetHeight() << "," << npu_idims[i].GetWidth()
            << "}";
    npu_idatasizes_[i] = npu_idims[i].GetNumber() * npu_idims[i].GetChannel() *
                         npu_idims[i].GetHeight() * npu_idims[i].GetWidth();
    CHECK_EQ(cpu_itensor->dims().production(), npu_idatasizes_[i]);
    npu_itensors_[i].reset(new hiai::AiTensor);
    npu_itensors_[i]->Init(&(npu_idims[i]));
  }
  for (size_t i = 0; i < npu_odims.size(); ++i) {
    auto cpu_otensor = param.outputs[i].second;
    CHECK(cpu_otensor);
    VLOG(3) << "[NPU] CPU output dims[" << i << "]: " << cpu_otensor->dims();
    VLOG(3) << "[NPU] NPU output dims[" << i << "]: {"
            << npu_odims[i].GetNumber() << "," << npu_odims[i].GetChannel()
            << "," << npu_odims[i].GetHeight() << "," << npu_odims[i].GetWidth()
            << "}";
    npu_odatasizes_[i] = npu_odims[i].GetNumber() * npu_odims[i].GetChannel() *
                         npu_odims[i].GetHeight() * npu_odims[i].GetWidth();
    if (cpu_otensor->dims().production() != npu_odatasizes_[i]) {
      cpu_otensor->Resize({npu_odims[i].GetNumber(),
                           npu_odims[i].GetChannel(),
                           npu_odims[i].GetHeight(),
                           npu_odims[i].GetWidth()});
    }
    npu_otensors_[i].reset(new hiai::AiTensor);
    npu_otensors_[i]->Init(&(npu_odims[i]));
  }
}

void GraphCompute::Run() {
  auto& param = this->Param<param_t>();

  // Check whether the data sizes of NPU input tensors are the same as
  // CPU's, and copy the data of CPU input tensors to NPU's.
  CHECK_EQ(param.inputs.size(), npu_itensors_.size());
  CHECK_EQ(param.outputs.size(), npu_otensors_.size());
  for (size_t i = 0; i < param.inputs.size(); ++i) {
    auto cpu_itensor = param.inputs[i].second;
    CHECK(cpu_itensor);
    CHECK_EQ(cpu_itensor->dims().production(), npu_idatasizes_[i]);
    std::memcpy(static_cast<float*>(npu_itensors_[i]->GetBuffer()),
                cpu_itensor->data<float>(),
                sizeof(float) * static_cast<size_t>(npu_idatasizes_[i]));
  }

  // Run HiAI model with model name
  std::string key = "model_name";  // Note: key seems must be model_name
  model_context_.AddPara(key, model_name_);
  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  int istamp;
  auto start_time = GetCurrentUS();
  CHECK_EQ(hiai::AI_SUCCESS,
           model_client_->Process(
               model_context_, npu_itensors_, npu_otensors_, 1000, istamp));
  VLOG(3) << "[NPU] Process cost " << GetCurrentUS() - start_time << " us";

  // Check whether the data sizes of NPU output tensors are the same as
  // CPU's, and copy the data of NPU output tensors to CPU's.
  for (size_t i = 0; i < param.outputs.size(); ++i) {
    auto cpu_otensor = param.outputs[i].second;
    CHECK(cpu_otensor);
    CHECK_EQ(cpu_otensor->dims().production(), npu_odatasizes_[i]);
    std::memcpy(cpu_otensor->mutable_data<float>(),
                static_cast<float*>(npu_otensors_[i]->GetBuffer()),
                sizeof(float) * static_cast<size_t>(npu_odatasizes_[i]));
  }
}

}  // namespace npu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(graph_op,
                     kNPU,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::npu::GraphCompute,
                     def)
    .BindInput("Inputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindInput("Weight", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Outputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
