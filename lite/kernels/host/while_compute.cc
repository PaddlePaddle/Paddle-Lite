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

#include "lite/kernels/host/while_compute.h"
#include <unordered_map>
#include <utility>

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

StepExecutor::StepExecutor(cpp::BlockDesc *block_desc,
                           Scope *scope,
                           const std::vector<std::string> &valid_places)
    : block_desc_(block_desc), scope_(scope) {
  for (auto &valid_place : valid_places) {
    auto parts = Split(valid_place, "/");
    CHECK_EQ(parts.size(), 3);
    TargetType target = static_cast<TargetType>(std::atoi(parts[0].c_str()));
    PrecisionType precision =
        static_cast<PrecisionType>(std::atoi(parts[1].c_str()));
    DataLayoutType layout =
        static_cast<DataLayoutType>(std::atoi(parts[2].c_str()));
    valid_places_.push_back(Place(target, precision, layout));
  }
}

void StepExecutor::Build() {
  for (int op_idx = 0; op_idx < block_desc_->OpsSize(); op_idx++) {
    auto op_desc = block_desc_->GetOp<cpp::OpDesc>(op_idx);
    CHECK(op_desc);
    std::string op_type = op_desc->Type();
    if (op_type == "feed" || op_type == "fetch") continue;
    // Create op and pick up the best kernel
    auto op = LiteOpRegistry::Global().Create(op_type);
    op->Attach(*op_desc, scope_);
    std::unique_ptr<KernelBase> picked_kernel;
    if (op_desc->HasAttr(kKernelTypeAttr)) {
      // Create op and pick up the best kernel according to the
      // kKernelTypeAttr attribute
      auto kernel_type = op_desc->GetAttr<std::string>(kKernelTypeAttr);
      std::string alias;
      Place place;
      KernelBase::ParseKernelType(kernel_type, &op_type, &alias, &place);
      VLOG(3) << "Found the attr '" << kKernelTypeAttr << "': " << kernel_type
              << " for " << op_type;
      auto kernels = op->CreateKernels({place});
      CHECK_GT(kernels.size(), 0) << "No kernels found for " << op_type;
      auto it = std::find_if(
          kernels.begin(), kernels.end(), [&](std::unique_ptr<KernelBase> &it) {
            return it->alias() == alias;
          });
      CHECK(it != kernels.end());
      picked_kernel = std::move(*it);
    } else {
      VLOG(3) << "The attr '" << kKernelTypeAttr
              << "' not found, pick the best kernel for " << op_type;
      auto kernels = op->CreateKernels(valid_places_);
      CHECK_GT(kernels.size(), 0) << "No kernels found for " << op_type;
      // Obtain the precision types of input arguments whose types are tensors.
      std::unordered_map<std::string, PrecisionType> in_types;
      auto in_names = op->op_info()->input_names();
      for (auto &in_name : in_names) {
        auto in_var = scope_->FindVar(in_name);
        if (in_var->IsType<Tensor>()) {
          auto in_precision = in_var->GetMutable<Tensor>()->precision();
          if (in_precision != PRECISION(kUnk)) {
            in_types[in_name] = in_precision;
          }
        }
      }
      // Pick up the best kernel according to the precision types of input
      // arguments and valid places
      core::KernelPickFactor kernel_pick_factors;
      kernel_pick_factors.ConsiderTarget();
      kernel_pick_factors.ConsiderPrecision();
      kernel_pick_factors.ConsiderDataLayout();
      CHECK(kernel_pick_factors.any_factor_considered())
          << "kernel_pick_factors should be specified first";
      float highest_score = 0;
      for (auto &&kernel : kernels) {
        float score = KernelGrade(*op->op_info(),
                                  *kernel,
                                  valid_places_,
                                  in_types,
                                  {},
                                  in_names,
                                  {},
                                  kernel_pick_factors);
        VLOG(4) << "kernel->summary():" << kernel->summary()
                << " score:" << score;
        if (score > highest_score) {
          picked_kernel = std::move(kernel);
          highest_score = score;
        }
      }
    }
    picked_kernel->SetContext(
        ContextScheduler::Global().NewContext(picked_kernel->target()));
    insts_.emplace_back(std::move(op), std::move(picked_kernel));
    insts_.back().Run();
  }
}

void StepExecutor::Run() {
  if (!insts_.empty()) {
    for (auto &inst : insts_) {
      auto op_type = inst.op()->op_info()->Type();
      if (op_type == "feed" || op_type == "fetch") continue;
      inst.Run();
    }
  } else {
    Build();
  }
}

void WhileCompute::PrepareForRun() {
  auto &param = this->Param<param_t>();
  executor_ = std::make_shared<StepExecutor>(
      param.sub_block, param.scope, param.valid_places);
}

void WhileCompute::Run() {
  auto &param = this->Param<param_t>();
  while (param.cond->data<bool>()[0]) {
    executor_->Run();
  }
}

}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    while, kHost, kAny, kAny, paddle::lite::kernels::host::WhileCompute, def)
    .BindInput("X",
               {LiteType::GetTensorListTy(
                   TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .BindInput("Condition",
               {LiteType::GetTensorTy(
                   TARGET(kHost), PRECISION(kBool), DATALAYOUT(kAny), -1)})
    .BindOutput("Out",
                {LiteType::GetTensorListTy(
                    TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .BindOutput("StepScopes",
                {LiteType::GetTensorTy(
                    TARGET(kHost), PRECISION(kAny), DATALAYOUT(kAny), -1)})
    .Finalize();
