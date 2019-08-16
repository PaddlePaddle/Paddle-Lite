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

#include <iostream>
#include "../test_helper.h"
#include "../test_include.h"
#include "operators/mul_op.h"

#define a(i, j) a[(i)*lda + (j)]
#define b(i, j) b[(i)*ldb + (j)]
#define c(i, j) c[(i)*ldc + (j)]

namespace paddle_mobile {
using framework::AttributeMap;
using framework::DDim;
using framework::Scope;
using framework::make_ddim;
template <typename I, typename O>
int TestMulOP() {
  int32_t m = 1024;
  int32_t n = 1024;
  int32_t k = 1024;
  int32_t lda = k;
  int32_t ldb = n;
  int32_t ldc = n;
  DDim inputA_shape = make_ddim({m, k});
  DDim inputB_shape = make_ddim({k, n});
  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<Scope>();
  inputs["X"] = std::vector<std::string>({"inputA"});
  inputs["Y"] = std::vector<std::string>({"inputB"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto inputA_var = scope.get()->Var("inputA");
  auto inputA = inputA_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<I>(inputA, inputA_shape, -127, 127);
  auto inputB_var = scope.get()->Var("inputB");
  auto inputB = inputB_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<I>(inputB, inputB_shape, -127, 127);

  auto output_var = scope.get()->Var("output");
  AttributeMap attrs;
  attrs["x_num_col_dims"].Set<int>(1);
  attrs["y_num_col_dims"].Set<int>(1);
  auto *op = new operators::MulOp<CPU, float>("mul", inputs, outputs, attrs,
                                              scope.get());
  op->InferShape();
  op->Run();
  auto output = output_var->template Get<framework::LoDTensor>();
  const O *output_data = output->data<O>();
  // compare
  O *c = static_cast<O *>(memory::Alloc(sizeof(O) * m * n));
  I *a = inputA->data<I>();
  I *b = inputB->data<I>();
  for (int32_t i = 0; i < m; ++i) {
    for (int32_t j = 0; j < n; ++j) {
      O r = 0;
      for (int32_t p = 0; p < k; p++) {
        r += static_cast<O>(a(i, p)) * static_cast<O>(b(p, j));
      }
      c(i, j) = r;
    }
  }

  int32_t eq = 0;
  int32_t neq = 0;
  for (int32_t i = 0; i < m * n; ++i) {
    PADDLE_MOBILE_ENFORCE(
        output_data[i] == c[i], "output[%d] = %d, output_cmp[%d] = %d", i,
        static_cast<int32_t>(output_data[i]), i, static_cast<int32_t>(c[i]));
    if (output_data[i] == c[i]) {
      ++eq;
    } else {
      ++neq;
    }
  }
  std::cout << "mnk=" << m << " " << n << " " << k << "   eq=" << eq
            << " neq=" << neq << std::endl;
  delete op;
  return 0;
}
}  // namespace paddle_mobile

int main() {
  paddle_mobile::PaddleMobile<paddle_mobile::CPU> paddle_mobile;
  paddle_mobile.SetThreadNum(4);
  paddle_mobile::TestMulOP<int8_t, int32_t>();
  paddle_mobile::TestMulOP<float, float>();
  return 0;
}
