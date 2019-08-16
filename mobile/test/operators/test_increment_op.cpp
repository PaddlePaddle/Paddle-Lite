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
#include "operators/increment_op.h"

namespace paddle_mobile {

template <typename T>
void Increment(const framework::Tensor *input, framework::Tensor *out,
               int step) {
  auto input_data = input->data<T>();
  auto out_data = out->data<T>();
  *out_data = *input_data + step;
}

int TestIncrementOp(const std::vector<int> input_shape, int step) {
  framework::DDim input_dims = framework::make_ddim(input_shape);
  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["X"] = std::vector<std::string>({"inputX"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto x_var = scope.get()->Var("inputX");
  auto x = x_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(x, input_dims, 0, 100);

  auto output_var = scope.get()->Var("output");
  framework::AttributeMap attrs;
  attrs["step"].Set<int>(step);

  auto *op = new operators::IncrementOp<CPU, float>(
      "increment", inputs, outputs, attrs, scope.get());

  op->InferShape();
  op->Init();
  op->Run();

  auto output = output_var->template Get<framework::LoDTensor>();
  framework::Tensor output_cmp;
  float *output_cmp_data = output_cmp.mutable_data<float>(output->dims());
  Increment<float>(x, &output_cmp, step);

  const float *output_data = output->data<float>();
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
}
}  // namespace paddle_mobile

int main() {
  paddle_mobile::TestIncrementOp({1}, 4);
  paddle_mobile::TestIncrementOp({1}, 10);
  DLOG << "test increment op pass.";
  return 0;
}
