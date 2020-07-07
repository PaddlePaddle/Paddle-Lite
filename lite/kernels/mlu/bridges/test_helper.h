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
#include <random>
#include <string>
#include <vector>
#include "lite/core/op_lite.h"
#include "lite/kernels/mlu/bridges/utility.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace mlu {

template <typename T>
std::shared_ptr<T> CreateOp(const cpp::OpDesc& opdesc, lite::Scope* scope) {
  auto op = std::make_shared<T>(opdesc.Type());
  op->SetValidPlaces(
      {Place{TARGET(kHost), PRECISION(kFloat)},
       Place{TARGET(kX86), PRECISION(kFloat)},
       Place{TARGET(kMLU), PRECISION(kFloat), DATALAYOUT(kNHWC)}});
  CHECK(op->Attach(opdesc, scope));
  CHECK(op->CheckShape());
  CHECK(op->InferShape());
  return op;
}

// T is the target data type
// R is the range data type, e.g. int, half
template <typename T, typename R = float>
void FillTensor(Tensor* x,
                T lower = static_cast<T>(-2),
                T upper = static_cast<T>(2)) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  T* x_data = x->mutable_data<T>();
  for (int i = 0; i < x->dims().production(); ++i) {
    auto r = uniform_dist(rng) * (upper - lower) + lower;
    x_data[i] = static_cast<T>(static_cast<R>(r));
  }
}

void LaunchOp(const std::shared_ptr<lite::OpLite> op,
              const std::vector<std::string>& input_var_names,
              const std::vector<std::string>& output_var_names,
              cnmlDataOrder_t order = CNML_NHWC);

}  // namespace mlu
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle
