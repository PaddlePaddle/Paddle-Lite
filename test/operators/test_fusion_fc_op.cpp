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
#include <type_traits>
#include "../test_helper.h"
#include "../test_include.h"
#include "framework/operator.h"
#include "operators/fusion_fc_op.h"

#define a(i, j) a[(i)*lda + (j)]
#define b(i, j) b[(i)*ldb + (j)]
#define c(i, j) c[(i)*ldc + (j)]

namespace paddle_mobile {
using framework::AttributeMap;
using framework::DDim;
using framework::Scope;
using framework::make_ddim;

int32_t qadd_int32(int32_t l, int32_t r) {
  int64_t res = static_cast<int64_t>(l) + static_cast<int64_t>(r);
  if (res > std::numeric_limits<int32_t>::max())
    return std::numeric_limits<int32_t>::max();
  else if (res < std::numeric_limits<int32_t>::min())
    return std::numeric_limits<int32_t>::min();
  else
    return static_cast<int32_t>(res);
}

// round to zero
float round2zero(float v) {
  float res;
  if (v > 0)
    res = std::floor(v);
  else if (v < 0)
    res = std::ceil(v);
  return res;
}

int8_t qscale_int32(int32_t v, float scale) {
  float res = static_cast<float>(v) * scale;
  res = round2zero(res);
  if (res > 127)
    return static_cast<int8_t>(127);
  else if (res < -127)
    return static_cast<int8_t>(-127);
  else
    return static_cast<int8_t>(res);
}

template <typename T, typename S>
int TestFcOP() {
  int32_t m = 377;
  int32_t n = 1363;
  int32_t k = 577;
  int32_t lda = k;
  int32_t ldb = n;
  int32_t ldc = n;
  DDim inputA_shape = make_ddim({m, k});
  DDim inputB_shape = make_ddim({k, n});
  DDim bias_shape = make_ddim({n});
  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<Scope>();
  inputs["X"] = std::vector<std::string>({"inputA"});
  inputs["Y"] = std::vector<std::string>({"inputB"});
  inputs["Z"] = std::vector<std::string>({"bias"});
  inputs["Scale"] = std::vector<std::string>({"scale"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto inputA_var = scope.get()->Var("inputA");
  auto inputA = inputA_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<T>(inputA, inputA_shape, -127, 127);
  auto inputB_var = scope.get()->Var("inputB");
  auto inputB = inputB_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<T>(inputB, inputB_shape, -127, 127);
  auto bias_var = scope.get()->Var("bias");
  auto bias = bias_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<S>(bias, bias_shape, -127, 127);

  auto scale_var = scope.get()->Var("scale");
  auto scale = scale_var->template GetMutable<framework::LoDTensor>();
  scale->Resize(framework::make_ddim({1}));
  float scale_v = 0.000828f;
  scale->mutable_data<float>()[0] = scale_v;

  auto output_var = scope.get()->Var("output");
  AttributeMap attrs;
  attrs["x_num_col_dims"].Set<int>(1);
  attrs["y_num_col_dims"].Set<int>(1);
  attrs["axis"].Set<int>(1);
  operators::OperatorBase<CPU> *op = nullptr;
  op = new operators::FusionFcOp<CPU, T>("fusion_fc", inputs, outputs, attrs,
                                         scope.get());
  op->InferShape();
  op->Run();
  auto output = output_var->template Get<framework::LoDTensor>();
  const T *output_data = output->data<T>();
  // compare
  T *c = static_cast<T *>(memory::Alloc(sizeof(T) * m * n));
  T *a = inputA->data<T>();
  T *b = inputB->data<T>();
  S *bias_data = bias->data<S>();
  for (int32_t i = 0; i < m; ++i) {
    for (int32_t j = 0; j < n; ++j) {
      S bias_v = bias_data[j];
      if (std::is_same<T, int8_t>::value) {
        int32_t r = 0;
        for (int32_t p = 0; p < k; p++) {
          r += static_cast<int32_t>(a(i, p)) * static_cast<int32_t>(b(p, j));
        }
        r = qadd_int32(r, bias_v);
        c(i, j) = qscale_int32(r, scale_v);
      } else {
        T r = 0;
        for (int32_t p = 0; p < k; p++) {
          r += a(i, p) * b(p, j);
        }
        r += bias_v;
        c(i, j) = r;
      }
    }
  }

  int32_t eq = 0;
  int32_t neq = 0;
  for (int32_t i = 0; i < m * n; ++i) {
    PADDLE_MOBILE_ENFORCE(output_data[i] == c[i],
                          "The execution of test_fusion_fc_op is failed!");
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
  paddle_mobile::TestFcOP<float, float>();
  return 0;
}
