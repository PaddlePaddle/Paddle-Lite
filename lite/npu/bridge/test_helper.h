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

#include <memory>
#include <string>
#include <vector>
#include "lite/core/op_lite.h"

namespace paddle {
namespace lite {
namespace npu {
namespace bridge {

template <typename T>
std::shared_ptr<T> CreateOp(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto op = std::make_shared<T>(opdesc.Type());
  op->SetValidPlaces({Place{TARGET(kHost), PRECISION(kFloat)},
                      Place{TARGET(kARM), PRECISION(kFloat)}});
  CHECK(op->Attach(opdesc, scope));
  CHECK(op->CheckShape());
  CHECK(op->InferShape());
  return op;
}

void LauchOp(const std::shared_ptr<lite::OpLite> op,
             const std::vector<std::string>& input_var_names,
             const std::vector<std::string>& output_var_names);

}  // namespace bridge
}  // namespace npu
}  // namespace lite
}  // namespace paddle
