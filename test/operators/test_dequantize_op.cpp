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

#include "../test_helper.h"
#include "../test_include.h"
#include "operators/dequantize_op.h"

namespace paddle_mobile {

void dequantize(const Tensor* input, const float scale, Tensor* output) {
  const int32_t* x = input->data<const int32_t>();
  float* y = output->mutable_data<float>();
  size_t size = output->numel();
  for (size_t i = 0; i < size; ++i) {
    y[i] = x[i] * scale;
  }
}

int TestDequqntizeOp() {
  framework::DDim dim = framework::make_ddim({1, 3, 224, 224});

  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["X"] = std::vector<std::string>({"input"});
  inputs["Scale"] = std::vector<std::string>({"scale"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<int32_t>(input, dim, -1000, 1000);

  auto scale_var = scope.get()->Var("scale");
  auto scale = scale_var->template GetMutable<framework::LoDTensor>();
  scale->Resize(framework::make_ddim({1}));
  scale->mutable_data<float>()[0] = 1.27;

  auto output_var = scope.get()->Var("output");
  framework::AttributeMap attrs;
  attrs["weight_scale"].Set<float>(1.74);

  auto* op = new operators::DequantizeOp<CPU, float>(
      "dequantize", inputs, outputs, attrs, scope.get());
  op->InferShape();
  op->Run();
  auto output = output_var->template Get<framework::LoDTensor>();
  const float* output_data = output->data<float>();

  framework::Tensor output_cmp;
  output_cmp.Resize(dim);
  float dequant_scale = 1.27 / 1.74;
  dequantize(input, dequant_scale, &output_cmp);
  const float* output_cmp_data = output_cmp.data<float>();
  for (int i = 0; i < output->numel(); ++i) {
    PADDLE_MOBILE_ENFORCE(output_data[i] == output_cmp_data[i],
                          "output[%d] = %.6f, output_cmp[%d] = %.6f", i,
                          output_data[i], i, output_cmp_data[i]);
  }
  delete op;
  return 0;
}

}  // namespace paddle_mobile

int main() { return paddle_mobile::TestDequqntizeOp(); }
