// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include <vector>
#include "lite/core/kernel.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace xpu {

class SetValueCompute : public KernelLite<TARGET(kXPU), PRECISION(kAny)> {
 public:
  using param_t = operators::SetValueParam;
  virtual ~SetValueCompute() = default;
  void PrepareForRun() override;
  virtual void Run();
  void SetValue(const std::vector<int64_t>& starts,
                const std::vector<int64_t>& ends,
                const std::vector<int64_t>& steps);

 private:
  XPUScratchPadGuard value_guard_;
};

static inline std::vector<int64_t> GetDataFromTensorList(
    const std::vector<const lite::Tensor*>& tensor_list) {
  std::vector<int64_t> vec_new_data;
  for (size_t i = 0; i < tensor_list.size(); ++i) {
    auto tensor = tensor_list[i];
    CHECK_EQ(tensor->dims(), DDim({1})) << "shape of dim tensor should be [1]";
    vec_new_data.push_back(static_cast<int64_t>(*tensor->data<int>()));
  }
  return vec_new_data;
}

}  // namespace xpu
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
