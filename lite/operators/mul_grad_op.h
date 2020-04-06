// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <string>
#include <vector>
#include "lite/core/kernel.h"
#include "lite/core/op_lite.h"
#include "lite/core/scope.h"
#include "lite/operators/op_params.h"
#include "lite/utils/all.h"

namespace paddle {
namespace lite {
namespace operators {

class MulGradOpLite : public OpLite {
 public:
  MulGradOpLite() {}

  explicit MulGradOpLite(const std::string &type) : OpLite(type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  void AttachKernel(KernelBase *kernel) override { kernel->SetParam(param_); }

  bool AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) override;

  std::string DebugString() const override { return "mul_grad"; }

 private:
  mutable MulGradParam param_;
};

std::vector<int64_t> flatten_2d(DDim dims, int num_col_dims) {
  std::vector<int64_t> flatten_dims{1, 1};
  for (int i = 0; i < dims.size(); i++) {
    if (i < num_col_dims) {
      flatten_dims[0] *= dims[i];
    } else {
      flatten_dims[1] *= dims[i];
    }
  }
  return flatten_dims;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle
