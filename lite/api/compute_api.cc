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

#include "compute_api.h"  // NOLINT
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/context.h"
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/op_registry.h"
#include "lite/core/program.h"
#include "log_lite.h"  // NOLINT

namespace paddle {
namespace lite_api {

class InstructionWrapper {
 public:
  InstructionWrapper(
      std::shared_ptr<lite::OpLite>& op,                          // NOLINT
      std::vector<std::unique_ptr<lite::KernelBase>>& kernels) {  // NOLINT
    op_ = op;
    for (auto& kernel : kernels) {
      kernels_.emplace_back(std::move(kernel));
    }
  }

  lite::OpLite* get_op() { return op_.get(); }

  lite::KernelBase* get_kernel() {
    if (kernel_idx > kernels_.size()) {
      LOGF("Error! kernel index > kernel size\n");
    }
    return kernels_[kernel_idx].get();
  }

  void set_kernel_idx(int idx) { kernel_idx = idx; }

  ~InstructionWrapper() = default;

 private:
  std::shared_ptr<lite::OpLite> op_;
  std::vector<std::unique_ptr<lite::KernelBase>> kernels_;
  int kernel_idx{0};
};

void ComputeEngine<TARGET(kARM)>::env_init(PowerMode power_mode, int threads) {
  lite::DeviceInfo::Init();
  lite::DeviceInfo::Global().SetRunMode(power_mode, threads);
}

bool ComputeEngine<TARGET(kARM)>::CreateOperator(const char* op_type,
                                                 PrecisionType precision,
                                                 DataLayoutType layout) {
  auto op = lite::LiteOpRegistry::Global().Create(op_type);
  LCHECK(op, "no Op found for %s\n", op_type);
  LOGI("Create %s Operator Success\n", op_type);
  lite_api::Place place(TARGET(kARM), precision, layout);
  auto kernels = op->CreateKernels({place});
  LCHECK_GT(kernels.size(), 0, "no kernel found for: %s\n", op_type);
  LOGI("Create %s kernel Success\n", op_type);
  instruction_ = new InstructionWrapper(op, kernels);
  return true;
}

// param must set input and output
void ComputeEngine<TARGET(kARM)>::SetParam(ParamBase* param) {
  delete static_cast<lite::operators::ParamBase*>(param_);
  // generate raw param
  param_ = param->AttachRawParam();
  auto* ins = static_cast<InstructionWrapper*>(instruction_);
  // pick kernel
  ins->set_kernel_idx(param->GetKernelIndex());
  // get raw kernel and op
  auto* kernel = ins->get_kernel();
  LCHECK(kernel, "SetParam, pick kernel error\n");
  auto* op = ins->get_op();
  // set context
  std::unique_ptr<lite::KernelContext> ctx(new lite::KernelContext);
  kernel->SetContext(std::move(ctx));
  op->SetParam(static_cast<lite::operators::ParamBase*>(param_));
  op->CheckShape();
  op->AttachKernel(kernel);
  LOGI("SetParam Success\n");
}

void ComputeEngine<TARGET(kARM)>::Launch() {
  auto* ins = static_cast<InstructionWrapper*>(instruction_);
  auto* kernel = ins->get_kernel();
  LCHECK(kernel, "Launch, pick kernel error\n");
  auto* op = ins->get_op();
  op->InferShapeImpl();
  kernel->Launch();
  LOGI("Run Success\n");
}

ComputeEngine<TARGET(kARM)>::~ComputeEngine() {
  delete static_cast<InstructionWrapper*>(instruction_);
  delete static_cast<lite::operators::ParamBase*>(param_);
  instruction_ = nullptr;
  param_ = nullptr;
}

}  // namespace lite_api
}  // namespace paddle
