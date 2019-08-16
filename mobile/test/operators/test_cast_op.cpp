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

#include "../test_include.h"
#include "operators/cast_op.h"

namespace paddle_mobile {

template <typename Itype, typename Otype>
void Cast(const framework::Tensor *X, framework::Tensor *Y) {
  const Itype *x = X->data<Itype>();
  Otype *y = Y->mutable_data<Otype>();

  for (int i = 0; i < X->numel(); ++i) {
    y[i] = static_cast<Otype>(x[i]);
  }
}

template <typename T>
int TypeInt() {}
template <>
int TypeInt<bool>() {
  return 0;
}
template <>
int TypeInt<int>() {
  return 2;
}
template <>
int TypeInt<int64_t>() {
  return 3;
}
template <>
int TypeInt<float>() {
  return 5;
}
template <>
int TypeInt<double>() {
  return 6;
}
template <>
int TypeInt<size_t>() {
  return 19;
}
template <>
int TypeInt<uint8_t>() {
  return 20;
}
template <>
int TypeInt<int8_t>() {
  return 21;
}

template <typename Itype, typename Otype>
int TestCastOp(const std::vector<int> input_shape) {
  framework::DDim dims = framework::make_ddim(input_shape);
  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["X"] = std::vector<std::string>({"input"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<Itype>(input, dims, static_cast<Itype>(-100),
                     static_cast<Itype>(100));

  auto output_var = scope.get()->Var("output");

  framework::AttributeMap attrs;
  attrs["in_dtype"].Set<int>(TypeInt<Itype>());
  attrs["out_dtype"].Set<int>(TypeInt<Otype>());
  auto *op = new operators::CastOp<CPU, float>("cast", inputs, outputs, attrs,
                                               scope.get());
  op->InferShape();
  op->Init();
  op->Run();

  auto output = output_var->template Get<framework::LoDTensor>();

  framework::Tensor output_cmp;
  Otype *output_cmp_data = output_cmp.mutable_data<Otype>(output->dims());
  Cast<Itype, Otype>(input, &output_cmp);

  const Otype *output_data = output->data<Otype>();
  for (int i = 0; i < output->numel(); ++i) {
    float gap = output_data[i] - output_cmp_data[i];
    if (std::abs(gap / (output_data[i] + 1e-5)) > 1e-3) {
      LOG(kLOG_INFO) << "output_data[" << i << "] = " << output_data[i]
                     << ", output_cmp_data[" << i
                     << "] = " << output_cmp_data[i];
      delete op;
      exit(1);
    }
  }
  delete op;
  return 0;
}

}  // namespace paddle_mobile

int main(int argc, char *argv[]) {
  TestCastOp<float, int>({1, 100});
  TestCastOp<float, int>({128, 100});

  TestCastOp<float, int64_t>({1, 100});
  TestCastOp<float, int64_t>({128, 100});

  TestCastOp<int, float>({1, 100});
  TestCastOp<int, float>({128, 100});

  TestCastOp<int64_t, float>({1, 100});
  TestCastOp<int64_t, float>({128, 100});
  return 0;
}
