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
#include "operators/batchnorm_op.h"

namespace paddle_mobile {

void BatchNorm(const framework::Tensor *X, const framework::Tensor *Mean,
               const framework::Tensor *Var, const framework::Tensor *Scale,
               const framework::Tensor *Bias, const float eps,
               framework::Tensor *Y) {
  const float *x = X->data<float>();
  const float *m = Mean->data<float>();
  const float *v = Var->data<float>();
  const float *s = Scale->data<float>();
  const float *b = Bias->data<float>();
  float *y = Y->mutable_data<float>();

  int batch_size = X->dims()[0];
  int channel = X->dims()[1];
  int hw = X->dims()[2] * X->dims()[3];

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int c = 0; c < channel; ++c) {
      float mean = m[c];
      float inv_var = 1.f / std::sqrt(v[c] + eps);
      float scale = s[c];
      float bias = b[c];
      const float *input = x + (batch * channel + c) * hw;
      float *output = y + (batch * channel + c) * hw;
      for (int j = 0; j < hw; ++j) {
        output[j] = scale * ((input[j] - mean) * inv_var) + bias;
      }
    }
  }
}

int TestBatchNormOp(const std::vector<int> input_shape) {
  framework::DDim dims = framework::make_ddim(input_shape);
  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["X"] = std::vector<std::string>({"input"});
  inputs["Mean"] = std::vector<std::string>({"mean"});
  inputs["Variance"] = std::vector<std::string>({"variance"});
  inputs["Scale"] = std::vector<std::string>({"scale"});
  inputs["Bias"] = std::vector<std::string>({"bias"});
  outputs["Y"] = std::vector<std::string>({"output"});

  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(input, dims, -100.0, 100.0);

  auto mean_var = scope.get()->Var("mean");
  auto mean = mean_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(mean, framework::make_ddim({input_shape[1]}), -10.0, 10.0);

  auto vari_var = scope.get()->Var("variance");
  auto vari = vari_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(vari, framework::make_ddim({input_shape[1]}), -10.0, 10.0);

  auto scale_var = scope.get()->Var("scale");
  auto scale = scale_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(scale, framework::make_ddim({input_shape[1]}), -10.0,
                     10.0);

  auto bias_var = scope.get()->Var("bias");
  auto bias = bias_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(bias, framework::make_ddim({input_shape[1]}), -10.0, 10.0);

  auto output_var = scope.get()->Var("output");

  float eps = 1e-6;
  framework::AttributeMap attrs;
  attrs["epsilon"].Set<float>(eps);
  attrs["momentum"].Set<float>(0.f);

  auto *op = new operators::BatchNormOp<CPU, float>(
      "batch_norm", inputs, outputs, attrs, scope.get());
  op->InferShape();
  op->Init();
  op->Run();

  auto output = output_var->template Get<framework::LoDTensor>();

  framework::Tensor output_cmp;
  float *output_cmp_data = output_cmp.mutable_data<float>(output->dims());
  BatchNorm(input, mean, vari, scale, bias, eps, &output_cmp);

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
  TestBatchNormOp({1, 1, 10, 10});
  TestBatchNormOp({1, 32, 100, 100});
  return 0;
}
