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
#include <string>
#include <vector>
#include "ai_ddk_lib/include/HiAiModelManagerService.h"
#include "lite/core/op_registry.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace npu {

void GraphCompute::PrepareForRun() {
  auto& ctx = this->ctx_->template As<NPUContext>();
  auto& param = this->Param<param_t>();

  exec_ = ctx.client(param.graph_name);
  CHECK(exec_);
  int ret =
      exec_->GetModelIOTensorDim(param.graph_name, npu_idims_, npu_odims_);
  CHECK_EQ(ret, hiai::AI_SUCCESS) << "[NPU] Get dims failed.";

  npu_itensors_.resize(npu_idims_.size());
  npu_otensors_.resize(npu_odims_.size());

  for (size_t i = 0; i < npu_idims_.size(); ++i) {
    npu_itensors_[i].reset(new hiai::AiTensor);
    npu_itensors_[i]->Init(&(npu_idims_[i]));
  }

  for (size_t i = 0; i < npu_odims_.size(); ++i) {
    npu_otensors_[i].reset(new hiai::AiTensor);
    npu_otensors_[i]->Init(&(npu_odims_[i]));
  }

  CHECK_EQ(param.output->dims().production(),
           npu_odims_[0].GetNumber() * npu_odims_[0].GetChannel() *
               npu_odims_[0].GetHeight() * npu_odims_[0].GetWidth());
}

bool GraphCompute::input_dims_changed() const {
  auto& param = this->Param<param_t>();
  // TODO(TJ): input change to vector
  CHECK(param.input);
  auto param_idims = param.input->dims();
  CHECK(!param_idims.empty());
  CHECK_EQ(npu_idims_.size(), 1);
  CHECK_EQ(param_idims.size(), 4);
  std::vector<int> idims{static_cast<int>(npu_idims_[0].GetNumber()),
                         static_cast<int>(npu_idims_[0].GetChannel()),
                         static_cast<int>(npu_idims_[0].GetHeight()),
                         static_cast<int>(npu_idims_[0].GetWidth())};
  for (size_t i = 0; i < 4; ++i) {
    if (param_idims[i] != idims[i]) {
      return true;
    }
  }
  return false;
}

void GraphCompute::Run() {
  CHECK(!input_dims_changed())
      << "When NPU is enabled, the input shape cloud not change yet.";
  auto& param = this->Param<param_t>();

  const auto* i_data = param.input->data<float>();
  auto* o_data = param.output->mutable_data<float>();

  // CHECK_EQ(param.inputs size, npu_itensors_.size());
  // TODO(TJ): vector
  std::memcpy(
      npu_itensors_[0]->GetBuffer(),
      i_data,
      sizeof(float) * static_cast<size_t>(param.input->dims().production()));
  std::string key = "graph_name";
  npu_context_.AddPara(key, param.graph_name);
  int istamp;
  CHECK_EQ(
      hiai::AI_SUCCESS,
      exec_->Process(npu_context_, npu_itensors_, npu_otensors_, 1000, istamp));
  auto* npu_obuffer = static_cast<float*>(npu_otensors_[0]->GetBuffer());

  std::memcpy(o_data,
              npu_obuffer,
              static_cast<size_t>(param.output->dims().production()));
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
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Output", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
