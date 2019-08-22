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

  exec_ = ctx.client(param.model_name);
  CHECK(exec_);
  int ret =
      exec_->GetModelIOTensorDim(param.model_name, npu_idims_, npu_odims_);
  CHECK_EQ(ret, hiai::AI_SUCCESS) << "[NPU] Get dims failed.";

  npu_itensors_.resize(npu_idims_.size());
  npu_otensors_.resize(npu_odims_.size());

  for (size_t i = 0; i < npu_idims_.size(); ++i) {
    VLOG(3) << "npu_idims[" << i << "]: " << npu_idims_[i].GetNumber() << ","
            << npu_idims_[i].GetChannel() << "," << npu_idims_[i].GetHeight()
            << "," << npu_idims_[i].GetWidth();
    VLOG(3) << "lite_idims[" << i << "]: " << param.inputs[i]->dims();
    CHECK_EQ(param.inputs[i]->dims().production(),
             npu_idims_[i].GetNumber() * npu_idims_[i].GetChannel() *
                 npu_idims_[i].GetHeight() * npu_idims_[i].GetWidth());
    npu_itensors_[i].reset(new hiai::AiTensor);
    npu_itensors_[i]->Init(&(npu_idims_[i]));
  }

  for (size_t i = 0; i < npu_odims_.size(); ++i) {
    VLOG(3) << "npu_odims[" << i << "]: " << npu_odims_[i].GetNumber() << ","
            << npu_odims_[i].GetChannel() << "," << npu_odims_[i].GetHeight()
            << "," << npu_odims_[i].GetWidth();
    VLOG(3) << "lite_odims[" << i << "]: " << param.outputs[i]->dims();
    auto out_size = npu_odims_[i].GetNumber() * npu_odims_[i].GetChannel() *
                    npu_odims_[i].GetHeight() * npu_odims_[i].GetWidth();
    if (param.outputs[i]->dims().production() != out_size) {
      param.outputs[i]->Resize({npu_odims_[i].GetNumber(),
                                npu_odims_[i].GetChannel(),
                                npu_odims_[i].GetHeight(),
                                npu_odims_[i].GetWidth()});
    }
    LOG(INFO) << param.outputs[i]->dims();
    npu_otensors_[i].reset(new hiai::AiTensor);
    npu_otensors_[i]->Init(&(npu_odims_[i]));
  }
}

bool GraphCompute::input_dims_changed() const {
  auto& param = this->Param<param_t>();
  CHECK_EQ(param.inputs.size(), npu_idims_.size());
  for (size_t i = 0; i < param.inputs.size(); ++i) {
    auto param_idims = param.inputs[i]->dims();
    CHECK(!param_idims.empty());
    CHECK_EQ(param_idims.size(), 4);
    std::vector<int> idims{static_cast<int>(npu_idims_[i].GetNumber()),
                           static_cast<int>(npu_idims_[i].GetChannel()),
                           static_cast<int>(npu_idims_[i].GetHeight()),
                           static_cast<int>(npu_idims_[i].GetWidth())};
    for (size_t i = 0; i < 4; ++i) {
      if (param_idims[i] != idims[i]) {
        return true;
      }
    }
  }

  return false;
}

void GraphCompute::Run() {
  CHECK(!input_dims_changed())
      << "When NPU is enabled, the input shape could not be changed yet.";
  auto& param = this->Param<param_t>();
  CHECK_EQ(param.inputs.size(), npu_itensors_.size());
  CHECK_EQ(param.outputs.size(), npu_otensors_.size());

  for (size_t i = 0; i < param.inputs.size(); ++i) {
    auto* itensor = param.inputs[i];
    CHECK(itensor);
    const auto* i_data = itensor->data<float>();
    std::memcpy(
        npu_itensors_[i]->GetBuffer(),
        i_data,
        sizeof(float) * static_cast<size_t>(itensor->dims().production()));
  }
  std::string key = "model_name";  // Note: key seems must be model_name
  npu_context_.AddPara(key, param.model_name);

  auto GetCurrentUS = []() -> double {
    struct timeval time;
    gettimeofday(&time, NULL);
    return 1e+6 * time.tv_sec + time.tv_usec;
  };
  int istamp;
  auto start_time = GetCurrentUS();
  CHECK_EQ(
      hiai::AI_SUCCESS,
      exec_->Process(npu_context_, npu_itensors_, npu_otensors_, 1000, istamp));
  LOG(INFO) << "[NPU] Process cost " << GetCurrentUS() - start_time << " us";

  for (size_t i = 0; i < param.outputs.size(); ++i) {
    auto* otensor = param.outputs[i];
    CHECK(otensor);
    auto* o_data = otensor->mutable_data<float>();
    auto* npu_obuffer = static_cast<float*>(npu_otensors_[i]->GetBuffer());

    std::memcpy(
        o_data,
        npu_obuffer,
        sizeof(float) * static_cast<size_t>(otensor->dims().production()));
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
    .BindOutput("Outputs", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
