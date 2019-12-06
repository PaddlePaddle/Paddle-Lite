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
#include <algorithm>
#include <memory>
#include <utility>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/program.h"
#include "lite/operators/conditional_block_op.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/basic_profiler.h"
#endif  // LITE_WITH_PROFILE
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/precision_profiler.h"
#endif

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

class CondExecutor {
  typedef std::shared_ptr<OpLite> OpPtr;

 public:
  CondExecutor(cpp::BlockDesc *block, Scope *scope, Place place)
      : scope_(scope), place_(place) {
    int32_t op_size = block->OpsSize();
    for (int32_t i = 0; i < op_size; ++i) {
      auto &op_desc = *block->template GetOp<cpp::OpDesc>(i);
      auto op_type = op_desc.Type();
      auto op_handler = lite::LiteOpRegistry::Global().Create(op_desc.Type());
      op_handler->Attach(op_desc, scope);

      auto hostplace = place_;
      hostplace.target = TARGET(kHost);
      auto kernels = op_handler->CreateKernels({place_, hostplace});
      CHECK_GT(kernels.size(), 0) << "cannot create kernel";
      op_handler->AttachKernel(kernels[0].get());
      op_handler->SetKernel(kernels);
      ops_of_block_.push_back(op_handler);
    }
  }

  void Run() {
    for (auto &op_handler : ops_of_block_) {
      op_handler->CheckShape();
      op_handler->InferShape();
#ifdef LITE_WITH_PROFILE
#ifdef LITE_WITH_PRECISION_PROFILE
      std::unique_ptr<KernelBase> kernel(op_handler->GetKernel());
      Instruction inst(op_handler, std::move(kernel));
#endif  // LITE_WITH_PRECISION_PROFILE
#endif  // LITE_WITH_PROFILE
      op_handler->Run();
#ifdef LITE_WITH_PROFILE
#ifdef LITE_WITH_PRECISION_PROFILE
      LITE_PRECISION_PROFILE(inst)
#endif  // LITE_WITH_PRECISION_PROFILE
#endif  // LITE_WITH_PROFILE
    }
  }

 private:
  Scope *scope_;
  Place place_;
  std::vector<OpPtr> ops_of_block_;
};

class ConditionalBlockCompute
    : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::ConditionalBlockParam;

  void PrepareForRun() override;
  void Run() override;

  virtual ~ConditionalBlockCompute() = default;

 private:
  std::shared_ptr<CondExecutor> executor_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
