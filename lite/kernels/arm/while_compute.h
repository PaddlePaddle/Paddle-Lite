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
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/operators/while_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

class StepExecutor {
  typedef std::shared_ptr<OpLite> OpPtr;

 public:
  StepExecutor(cpp::BlockDesc *block, Scope *scope, Place place)
      : scope_(scope), place_(place) {
    int32_t op_size = block->OpsSize();
    for (int32_t i = 0; i < op_size; ++i) {
      cpp::OpDesc *op_desc = block->template GetOp<cpp::OpDesc>(i);
      auto op_handler = lite::LiteOpRegistry::Global().Create(op_desc->Type());
      op_handler->Attach(*op_desc, scope);
      auto kernels = op_handler->CreateKernels({place_});
      CHECK_GT(kernels.size(), 0) << "cannot create kernel";
      op_handler->AttachKernel(kernels[0].get());
      ops_of_block_.push_back(op_handler);
    }
  }

  void Run() {
    for (auto &op_handler : ops_of_block_) {
      op_handler->InferShape();
      op_handler->Run();
    }
  }

 private:
  Scope *scope_;
  Place place_;
  std::vector<OpPtr> ops_of_block_;
};

class WhileCompute : public KernelLite<TARGET(kARM), PRECISION(kFloat)> {
 public:
  using param_t = operators::WhileParam;

  void Run() override;
  void PrepareForRun() override;

  virtual ~WhileCompute() = default;

 private:
  std::shared_ptr<StepExecutor> executor_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
