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

#include "lite/model_parser/flatbuffers/io.h"
#include <gtest/gtest.h>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace fbs {

namespace {
template <typename T>
void set_tensor(paddle::lite::Tensor* tensor,
                const std::vector<int64_t>& dims) {
  auto production =
      std::accumulate(begin(dims), end(dims), 1, std::multiplies<int64_t>());
  tensor->Resize(dims);
  std::vector<T> data;
  data.resize(production);
  for (int i = 0; i < production; ++i) {
    data[i] = i / 2.f;
  }
  std::memcpy(tensor->mutable_data<T>(), data.data(), sizeof(T) * data.size());
}
}  // namespace

TEST(CombinedParamsDesc, Scope) {
  /* --------- Save scope ---------- */
  Scope scope;
  std::vector<std::string> params_name({"var_0", "var_1"});
  // variable 0
  Variable* var_0 = scope.Var(params_name[0]);
  Tensor* tensor_0 = var_0->GetMutable<Tensor>();
  set_tensor<float>(tensor_0, std::vector<int64_t>({3, 2}));
  // variable 1
  Variable* var_1 = scope.Var(params_name[1]);
  Tensor* tensor_1 = var_1->GetMutable<Tensor>();
  set_tensor<int8_t>(tensor_1, std::vector<int64_t>({10, 1}));
  // Set combined parameters
  fbs::CombinedParamsDesc combined_param;
  std::set<std::string> params_set(params_name.begin(), params_name.end());
  SetCombinedParamsWithScope(scope, params_set, &combined_param);

  /* --------- Check scope ---------- */
  auto check_params = [&](const CombinedParamsDescReadAPI& desc) {
    Scope scope_l;
    SetScopeWithCombinedParams(&scope_l, desc);
    // variable 0
    Variable* var_l0 = scope_l.FindVar(params_name[0]);
    CHECK(var_l0);
    const Tensor& tensor_l0 = var_l0->Get<Tensor>();
    CHECK(TensorCompareWith(*tensor_0, tensor_l0));
    // variable 1
    Variable* var_l1 = scope_l.FindVar(params_name[1]);
    CHECK(var_l1);
    const Tensor& tensor_l1 = var_l1->Get<Tensor>();
    CHECK(TensorCompareWith(*tensor_1, tensor_l1));
  };
  check_params(combined_param);

  /* --------- Cache scope ---------- */
  std::vector<char> cache;
  cache.resize(combined_param.buf_size());
  std::memcpy(cache.data(), combined_param.data(), combined_param.buf_size());

  /* --------- View scope ---------- */
  check_params(CombinedParamsDescView(std::move(cache)));
}

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
