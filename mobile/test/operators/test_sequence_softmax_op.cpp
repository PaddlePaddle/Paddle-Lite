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
#include "operators/sequence_ops/sequence_softmax_op.h"

namespace paddle_mobile {

void SequenceSoftmax(const framework::LoDTensor *X, framework::LoDTensor *Y) {
  const float *x = X->data<float>();
  const auto &lod = X->lod().back();
  float *y = Y->mutable_data<float>();
  for (int batch = 0; batch < lod.size() - 1; ++batch) {
    int num_classes = lod[batch + 1] - lod[batch];
    size_t offset = lod[batch];
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
  Y->set_lod(X->lod());
}

int TestSequenceSoftmaxOp(const std::vector<int> &input_shape,
                          const std::vector<size_t> &input_lod) {
  framework::DDim dims = framework::make_ddim(input_shape);
  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["X"] = std::vector<std::string>({"input"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(input, dims, -100.0, 100.0);
  input->set_lod({input_lod});

  auto output_var = scope.get()->Var("output");

  framework::AttributeMap attrs;
  auto *op = new operators::SequenceSoftmaxOp<CPU, float>(
      "sequence_softmax", inputs, outputs, attrs, scope.get());

  op->InferShape();
  op->Init();
  op->Run();

  auto output = output_var->template Get<framework::LoDTensor>();

  framework::LoDTensor output_cmp;
  float *output_cmp_data = output_cmp.mutable_data<float>(output->dims());
  SequenceSoftmax(input, &output_cmp);

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
  TestSequenceSoftmaxOp({2, 1}, {0, 2});
  TestSequenceSoftmaxOp({100, 1}, {0, 3, 100});
  TestSequenceSoftmaxOp({100, 1}, {0, 50, 100});
  return 0;
}
