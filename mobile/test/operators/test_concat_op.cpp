/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <cstring>
#include <iostream>
#include <vector>
#include "../test_helper.h"
#include "../test_include.h"
#include "operators/concat_op.h"

namespace paddle_mobile {
using framework::AttributeMap;
using framework::DDim;
using framework::LoDTensor;
using framework::Scope;
using framework::make_ddim;

template <typename T>
void concat(const std::vector<LoDTensor> &input, LoDTensor *output, int axis) {
  int num = input.size();

  int rows = 1;
  auto dim_0 = input[0].dims();
  for (int i = 0; i < axis; ++i) {
    rows *= dim_0[i];
  }
  int out_rows = rows, out_cols = 0;

  std::vector<int> input_cols(input.size());
  for (int i = 0; i < num; ++i) {
    int t_cols = input[i].numel() / rows;
    out_cols += t_cols;
    input_cols[i] = t_cols;
  }

  // computation
  auto output_data = output->data<T>();
  int col_idx = 0;
  for (int j = 0; j < num; ++j) {
    int col_len = input_cols[j];
    auto input_data = input[j].data<T>();
    for (int k = 0; k < out_rows; ++k) {
      memcpy(output_data + k * out_cols + col_idx, input_data + k * col_len,
             sizeof(T) * col_len);
    }
    col_idx += col_len;
  }
}

template <typename T>
int TestConcatOP() {
  DDim inputA_shape = make_ddim({10, 4, 2, 2});
  DDim inputB_shape = make_ddim({20, 4, 2, 2});
  DDim inputC_shape = make_ddim({30, 4, 2, 2});
  DDim inputD_shape = make_ddim({40, 4, 2, 2});
  DDim output_shape = make_ddim({100, 4, 2, 2});
  int axis_v = 0;
  VariableNameMap inputs;
  VariableNameMap outputs;
  std::vector<LoDTensor> input_tensors;
  auto scope = std::make_shared<Scope>();
  inputs["X"] =
      std::vector<std::string>({"inputA", "inputB", "inputC", "inputD"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto inputA_var = scope.get()->Var("inputA");
  auto inputA = inputA_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<T>(inputA, inputA_shape, -127, 127);
  input_tensors.push_back(std::move(*inputA));

  auto inputB_var = scope.get()->Var("inputB");
  auto inputB = inputB_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<T>(inputB, inputB_shape, -127, 127);
  input_tensors.push_back(std::move(*inputB));

  auto inputC_var = scope.get()->Var("inputC");
  auto inputC = inputC_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<T>(inputC, inputC_shape, -127, 127);
  input_tensors.push_back(std::move(*inputC));

  auto inputD_var = scope.get()->Var("inputD");
  auto inputD = inputD_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<T>(inputD, inputD_shape, -127, 127);
  input_tensors.push_back(std::move(*inputD));

  auto output_var = scope.get()->Var("output");
  AttributeMap attrs;
  attrs["axis"].Set<int>(axis_v);

  auto *op = new operators::ConcatOp<CPU, float>("concat", inputs, outputs,
                                                 attrs, scope.get());
  op->InferShape();
  op->Run();
  auto output = output_var->template Get<framework::LoDTensor>();
  const T *output_data = output->data<T>();
  LoDTensor output_cmp;
  output_cmp.mutable_data<T>(output_shape);
  concat<T>(input_tensors, &output_cmp, axis_v);
  const T *output_cmp_data = output_cmp.data<T>();
  // compare
  int eq = 0;
  int neq = 0;
  for (int i = 0; i < output->numel(); ++i) {
    PADDLE_MOBILE_ENFORCE(output_data[i] == output_cmp_data[i],
                          "The execution of test_concat_op is failed!");
    if (output_data[i] == output_cmp_data[i]) {
      ++eq;
    } else {
      ++neq;
    }
  }
  std::cout << "eq = " << eq << ", neq = " << neq << std::endl;

  delete op;
  return 0;
}
}  // namespace paddle_mobile

int main() {
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  paddle_mobile.SetThreadNum(4);
  paddle_mobile::TestConcatOP<float>();
  paddle_mobile::TestConcatOP<int8_t>();
  return 0;
}
