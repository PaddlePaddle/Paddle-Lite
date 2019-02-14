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
#include "../test_include.h"
#include "operators/sequence_ops/sequence_expand_op.h"

namespace paddle_mobile {

int TestSequenceExpandOp(const framework::LoDTensor &input_x,
                         const framework::LoDTensor &input_y, int ref_level,
                         framework::LoDTensor *output) {
  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["X"] = std::vector<std::string>({"input_x"});
  inputs["Y"] = std::vector<std::string>({"input_y"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto input_x_var = scope.get()->Var("input_x");
  auto *x = input_x_var->template GetMutable<framework::LoDTensor>();
  x->Resize(input_x.dims());
  x->ShareDataWith(input_x);
  x->set_lod(input_x.lod());
  auto input_y_var = scope.get()->Var("input_y");
  auto *y = input_y_var->template GetMutable<framework::LoDTensor>();
  y->Resize(framework::make_ddim({0}));
  y->mutable_data<float>();
  y->set_lod(input_y.lod());

  auto output_var = scope.get()->Var("output");

  framework::AttributeMap attrs;
  attrs["ref_level"].Set<int>(0);

  auto *op = new operators::SequenceExpandOp<CPU, float>(
      "sequence_expand", inputs, outputs, attrs, scope.get());

  op->InferShape();
  op->Init();
  op->Run();

  auto *out = output_var->template Get<framework::LoDTensor>();
  output->Resize(out->dims());
  output->ShareDataWith(*out);
  output->set_lod(out->lod());
  delete op;
  return 0;
}

}  // namespace paddle_mobile

// namespace framework = paddle_mobile::framework;

int main(int argc, char *argv[]) {
  framework::LoDTensor input_x, input_y, output;
  // case 1
  {
    std::vector<float> data{1, 2, 3, 4};
    input_x.Resize(framework::make_ddim({4, 1}));
    float *in_data = input_x.mutable_data<float>();
    for (int i = 0; i < 4; ++i) in_data[i] = data[i];
    input_x.set_lod({{0, 2, 4}});
    input_y.set_lod({{0, 2, 4}, {0, 3, 6, 7, 8}});

    TestSequenceExpandOp(input_x, input_y, 0, &output);
    std::vector<float> expect_data{1, 2, 1, 2, 3, 4, 3, 4};
    std::vector<int> expect_lod{0, 2, 4, 6, 8};
    for (int i = 0; i < 5; ++i) {
      if (output.lod()[0][i] != expect_lod[i]) {
        std::cerr << "output_lod[" << i << "]: " << output.lod()[0][i]
                  << " != expect_lod[" << i << "]: " << expect_lod[i]
                  << std::endl;
        return 1;
      }
    }
    for (int i = 0; i < 8; ++i) {
      if (output.data<float>()[i] != expect_data[i]) {
        std::cerr << "output[" << i << "]: " << output.data<float>()[i]
                  << " != expect[" << i << "]: " << expect_data[i] << std::endl;
        return 1;
      }
    }
  }
  return 0;
}
