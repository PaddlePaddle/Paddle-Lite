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
#include "lite/model_parser/model_parser.h"

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
  tensor->set_persistable(true);
  std::vector<T> data;
  data.resize(production);
  for (int i = 0; i < production; ++i) {
    data[i] = i / 2.f;
  }
  std::memcpy(tensor->mutable_data<T>(), data.data(), sizeof(T) * data.size());
}
}  // namespace

#ifdef LITE_WITH_FLATBUFFERS_DESC
TEST(CombinedParamsDesc, Scope) {
  const std::string path{"io_test.params.fbs"};
  /* --------- Save scope ---------- */
  Scope scope;
  std::vector<std::string> param_names({"var_0", "var_1", "var_2"});
  // variable 0
  Variable* var_0 = scope.Var(param_names[0]);
  Tensor* tensor_0 = var_0->GetMutable<Tensor>();
  set_tensor<float>(tensor_0, std::vector<int64_t>({3, 2}));
  // variable 1
  Variable* var_1 = scope.Var(param_names[1]);
  Tensor* tensor_1 = var_1->GetMutable<Tensor>();
  set_tensor<int8_t>(tensor_1, std::vector<int64_t>({10, 1}));
  // variable 3
  Variable* var_2 = scope.Var(param_names[2]);
  Tensor* tensor_2 = var_2->GetMutable<Tensor>();
  set_tensor<int16_t>(tensor_2, std::vector<int64_t>({16, 1}));

  std::set<std::string> params_set(param_names.begin(), param_names.end());

  {
    model_parser::BinaryFileWriter writer{path};
    fbs::ParamSerializer serializer{&writer};
    serializer.ForwardWrite(scope, params_set);
  }

  // Set combined parameters
  fbs::CombinedParamsDesc combined_param;
  deprecated::SetCombinedParamsWithScope(scope, params_set, &combined_param);

  auto check_params = [&](const lite::Scope& scope) {
    // variable 0
    Variable* var_l0 = scope.FindVar(param_names[0]);
    CHECK(var_l0);
    const Tensor& tensor_l0 = var_l0->Get<Tensor>();
    CHECK(TensorCompareWith(*tensor_0, tensor_l0));
    // variable 1
    Variable* var_l1 = scope.FindVar(param_names[1]);
    CHECK(var_l1);
    const Tensor& tensor_l1 = var_l1->Get<Tensor>();
    CHECK(TensorCompareWith(*tensor_1, tensor_l1));
    // variable 2
    Variable* var_l2 = scope.FindVar(param_names[2]);
    CHECK(var_l2);
    const Tensor& tensor_l2 = var_l2->Get<Tensor>();
    CHECK(TensorCompareWith(*tensor_2, tensor_l2));
  };

  model_parser::Buffer buffer;

  /* --------- Scope ---------- */
  Scope scope_0;
  deprecated::SetScopeWithCombinedParams(&scope_0, combined_param);
  check_params(scope_0);

  /* --------- View scope ---------- */
  Scope scope_1;
  combined_param.CopyDataToBuffer(&buffer);
  CombinedParamsDescView combined_param_view(std::move(buffer));
  deprecated::SetScopeWithCombinedParams(&scope_1, combined_param_view);
  check_params(scope_1);

  /* --------- Stream scope ---------- */
  {
    Scope scope_2;
    LOG(INFO) << "Load params from file...";
    model_parser::BinaryFileReader reader(path);
    fbs::ParamDeserializer deserializer(&reader);
    deserializer.ForwardRead(&scope_2);
    check_params(scope_2);
  }

  {
    Scope scope_3;
    LOG(INFO) << "Load params from string buffer...";
    model_parser::BinaryFileReader file_reader(path);
    model_parser::Buffer buf(file_reader.length());
    file_reader.Read(buf.data(), file_reader.length());

    std::string str{static_cast<const char*>(buf.data()), buf.size()};

    model_parser::StringBufferReader reader(str);
    fbs::ParamDeserializer deserializer(&reader);
    deserializer.ForwardRead(&scope_3);
    check_params(scope_3);
  }
}
#endif  // LITE_WITH_FLATBUFFERS_DESC

}  // namespace fbs
}  // namespace lite
}  // namespace paddle
