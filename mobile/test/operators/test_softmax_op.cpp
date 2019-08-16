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

#include <math.h>
#include <limits>
#include "../test_include.h"
#include "operators/softmax_op.h"

namespace paddle_mobile {

void Softmax(const framework::Tensor *X, framework::Tensor *Y) {
  const framework::DDim &dims = X->dims();
  int batch_size = dims[0];
  int num_classes = dims[dims.size() - 1];
  int channels = X->numel() / batch_size / num_classes;
  const float *x = X->data<float>();
  float *y = Y->mutable_data<float>();

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int c = 0; c < channels; ++c) {
      size_t offset = (batch * channels + c) * num_classes;
      const float *input = x + offset;
      float *output = y + offset;
      float max = -std::numeric_limits<float>::max();
      for (int j = 0; j < num_classes; ++j) {
        max = (input[j] > max) ? input[j] : max;
      }
      float sum = 0.f;
      for (int j = 0; j < num_classes; ++j) {
        float tmp = expf(input[j] - max);
        sum += tmp;
        output[j] = tmp;
      }
      for (int j = 0; j < num_classes; ++j) {
        output[j] /= sum;
      }
    }
  }
}

int TestSoftmaxOp(const std::vector<int> input_shape) {
  framework::DDim dims = framework::make_ddim(input_shape);
  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["X"] = std::vector<std::string>({"input"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(input, dims, -100.0, 100.0);

  auto output_var = scope.get()->Var("output");

  framework::AttributeMap attrs;
  auto *op = new operators::SoftmaxOp<CPU, float>("softmax", inputs, outputs,
                                                  attrs, scope.get());
  op->InferShape();
  op->Init();
  op->Run();

  auto output = output_var->template Get<framework::LoDTensor>();

  framework::Tensor output_cmp;
  float *output_cmp_data = output_cmp.mutable_data<float>(output->dims());
  Softmax(input, &output_cmp);

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
  delete op;
  return 0;
}

}  // namespace paddle_mobile

int main(int argc, char *argv[]) {
  TestSoftmaxOp({128, 1000});
  TestSoftmaxOp({128, 10, 1000});
  return 0;
}
