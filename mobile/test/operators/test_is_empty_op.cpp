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
#include "operators/is_empty_op.h"

namespace paddle_mobile {

void IsEmpty(const framework::Tensor *input, framework::Tensor *out) {
  out->data<bool>()[0] = input->numel() == 0;
}

int TestIsEmptyOp(const std::vector<int> input_shape) {
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

  auto *op = new operators::IsEmptyOp<CPU, float>("is_empty", inputs, outputs,
                                                  attrs, scope.get());

  op->InferShape();
  op->Init();
  op->Run();

  auto output = output_var->template Get<framework::LoDTensor>();
  framework::Tensor output_cmp;
  bool *output_cmp_data = output_cmp.mutable_data<bool>(output->dims());
  IsEmpty(x, &output_cmp);

  const bool *output_data = output->data<bool>();
  for (int i = 0; i < output->numel(); ++i) {
    if (output_data[i] != output_cmp_data[i]) {
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
  paddle_mobile::TestIsEmptyOp({1, 3, 100, 100});
  paddle_mobile::TestIsEmptyOp({0});
  DLOG << "test is_empty op pass.";
  return 0;
}
