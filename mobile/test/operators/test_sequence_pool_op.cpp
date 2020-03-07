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
#include "operators/sequence_ops/sequence_pool_op.h"

namespace paddle_mobile {

int TestSequencePoolOp(const framework::LoDTensor &input_x,
                       const std::string pool_type,
                       framework::LoDTensor *output) {
  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["X"] = std::vector<std::string>({"input_x"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto input_x_var = scope.get()->Var("input_x");
  auto *x = input_x_var->template GetMutable<framework::LoDTensor>();
  x->Resize(input_x.dims());
  x->ShareDataWith(input_x);
  x->set_lod(input_x.lod());

  auto output_var = scope.get()->Var("output");

  framework::AttributeMap attrs;
  attrs["pooltype"].Set<std::string>(pool_type);

  auto *op = new operators::SequencePoolOp<CPU, float>(
      "sequence_pool", inputs, outputs, attrs, scope.get());

  op->InferShape();
  op->Init();
  op->Run();

  auto *out = output_var->template Get<framework::LoDTensor>();
  output->Resize(out->dims());
  output->ShareDataWith(*out);
  delete op;
  return 0;
}

}  // namespace paddle_mobile

// namespace framework = paddle_mobile::framework;

int main(int argc, char *argv[]) {
  framework::LoDTensor input_x, output;
  // case 1
  DLOG << "running max case 1";
  {
    std::vector<float> data{1, 2, 3, 4};
    input_x.Resize(framework::make_ddim({4, 1}));
    float *in_data = input_x.mutable_data<float>();
    for (int i = 0; i < 4; ++i) in_data[i] = data[i];
    input_x.set_lod({{0, 2, 4}});

    TestSequencePoolOp(input_x, "MAX", &output);
    std::vector<float> expect_data{2, 4};
    for (int i = 0; i < 2; ++i) {
      if (output.data<float>()[i] != expect_data[i]) {
        DLOG << "output[" << i << "]: " << output.data<float>()[i]
             << " != expect[" << i << "]: " << expect_data[i];
        return 1;
      }
    }
  }
  // case 2
  DLOG << "running max case 2";
  {
    std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    input_x.Resize(framework::make_ddim({data.size(), 1}));
    float *in_data = input_x.mutable_data<float>();
    for (int i = 0; i < data.size(); ++i) in_data[i] = data[i];
    input_x.set_lod({{0, 3, 10}});

    TestSequencePoolOp(input_x, "MAX", &output);
    std::vector<float> expect_data{3, 10};
    for (int i = 0; i < 2; ++i) {
      if (output.data<float>()[i] != expect_data[i]) {
        DLOG << "output[" << i << "]: " << output.data<float>()[i]
             << " != expect[" << i << "]: " << expect_data[i];
        return 1;
      }
    }
  }
  DLOG << "running max case 3";
  // case 3
  {
    std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8};
    input_x.Resize(framework::make_ddim({4, 2}));
    float *in_data = input_x.mutable_data<float>();
    for (int i = 0; i < data.size(); ++i) in_data[i] = data[i];
    input_x.set_lod({{0, 2, 4}});

    TestSequencePoolOp(input_x, "MAX", &output);
    std::vector<float> expect_data{3, 4, 7, 8};
    for (int i = 0; i < 4; ++i) {
      if (output.data<float>()[i] != expect_data[i]) {
        DLOG << "output[" << i << "]: " << output.data<float>()[i]
             << " != expect[" << i << "]: " << expect_data[i];
        return 1;
      }
    }
  }
  // case 4
  DLOG << "running max case 4";
  {
    std::vector<float> data{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    input_x.Resize(framework::make_ddim({4, 5}));
    float *in_data = input_x.mutable_data<float>();
    for (int i = 0; i < data.size(); ++i) in_data[i] = data[i];
    input_x.set_lod({{0, 2, 4}});

    TestSequencePoolOp(input_x, "MAX", &output);
    std::vector<float> expect_data{6, 7, 8, 9, 10, 16, 17, 18, 19, 20};
    for (int i = 0; i < 10; ++i) {
      if (output.data<float>()[i] != expect_data[i]) {
        DLOG << "output[" << i << "]: " << output.data<float>()[i]
             << " != expect[" << i << "]: " << expect_data[i];
        return 1;
      }
    }
  }
  // case 1
  DLOG << "running sum case 1";
  {
    std::vector<float> data{1, 2, 3, 4};
    input_x.Resize(framework::make_ddim({4, 1}));
    float *in_data = input_x.mutable_data<float>();
    for (int i = 0; i < 4; ++i) in_data[i] = data[i];
    input_x.set_lod({{0, 2, 4}});

    TestSequencePoolOp(input_x, "SUM", &output);
    std::vector<float> expect_data{3, 7};
    for (int i = 0; i < 2; ++i) {
      if (output.data<float>()[i] != expect_data[i]) {
        DLOG << "output[" << i << "]: " << output.data<float>()[i]
             << " != expect[" << i << "]: " << expect_data[i];
        return 1;
      }
    }
  }
  // case 2
  DLOG << "running sum case 2";
  {
    std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    input_x.Resize(framework::make_ddim({data.size(), 1}));
    float *in_data = input_x.mutable_data<float>();
    for (int i = 0; i < data.size(); ++i) in_data[i] = data[i];
    input_x.set_lod({{0, 3, 10}});

    TestSequencePoolOp(input_x, "SUM", &output);
    std::vector<float> expect_data{6, 49};
    for (int i = 0; i < 2; ++i) {
      if (output.data<float>()[i] != expect_data[i]) {
        DLOG << "output[" << i << "]: " << output.data<float>()[i]
             << " != expect[" << i << "]: " << expect_data[i];
        return 1;
      }
    }
  }
  // case 3
  DLOG << "running sum case 3";
  {
    std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8};
    input_x.Resize(framework::make_ddim({4, 2}));
    float *in_data = input_x.mutable_data<float>();
    for (int i = 0; i < data.size(); ++i) in_data[i] = data[i];
    input_x.set_lod({{0, 2, 4}});

    TestSequencePoolOp(input_x, "SUM", &output);
    std::vector<float> expect_data{4, 6, 12, 14};
    for (int i = 0; i < 4; ++i) {
      if (output.data<float>()[i] != expect_data[i]) {
        DLOG << "output[" << i << "]: " << output.data<float>()[i]
             << " != expect[" << i << "]: " << expect_data[i];
        return 1;
      }
    }
  }
  // case 4
  DLOG << "running sum case 4";
  {
    std::vector<float> data{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    input_x.Resize(framework::make_ddim({4, 5}));
    float *in_data = input_x.mutable_data<float>();
    for (int i = 0; i < data.size(); ++i) in_data[i] = data[i];
    input_x.set_lod({{0, 2, 4}});

    TestSequencePoolOp(input_x, "SUM", &output);
    std::vector<float> expect_data{7, 9, 11, 13, 15, 27, 29, 31, 33, 35};
    for (int i = 0; i < 10; ++i) {
      if (output.data<float>()[i] != expect_data[i]) {
        DLOG << "output[" << i << "]: " << output.data<float>()[i]
             << " != expect[" << i << "]: " << expect_data[i];
        return 1;
      }
    }
  }
  // case 1
  DLOG << "running first case 1";
  {
    std::vector<float> data{1, 2, 3, 4};
    input_x.Resize(framework::make_ddim({4, 1}));
    float *in_data = input_x.mutable_data<float>();
    for (int i = 0; i < 4; ++i) in_data[i] = data[i];
    input_x.set_lod({{0, 2, 4}});

    TestSequencePoolOp(input_x, "FIRST", &output);
    std::vector<float> expect_data{1, 3};
    for (int i = 0; i < 2; ++i) {
      if (output.data<float>()[i] != expect_data[i]) {
        DLOG << "output[" << i << "]: " << output.data<float>()[i]
             << " != expect[" << i << "]: " << expect_data[i];
        return 1;
      }
    }
  }
  // case 2
  DLOG << "running first case 2";
  {
    std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    input_x.Resize(framework::make_ddim({data.size(), 1}));
    float *in_data = input_x.mutable_data<float>();
    for (int i = 0; i < data.size(); ++i) in_data[i] = data[i];
    input_x.set_lod({{0, 3, 10}});

    TestSequencePoolOp(input_x, "FIRST", &output);
    std::vector<float> expect_data{1, 4};
    for (int i = 0; i < 2; ++i) {
      if (output.data<float>()[i] != expect_data[i]) {
        DLOG << "output[" << i << "]: " << output.data<float>()[i]
             << " != expect[" << i << "]: " << expect_data[i];
        return 1;
      }
    }
  }
  // case 3
  DLOG << "running first case 3";
  {
    std::vector<float> data{1, 2, 3, 4, 5, 6, 7, 8};
    input_x.Resize(framework::make_ddim({4, 2}));
    float *in_data = input_x.mutable_data<float>();
    for (int i = 0; i < data.size(); ++i) in_data[i] = data[i];
    input_x.set_lod({{0, 2, 4}});

    TestSequencePoolOp(input_x, "FIRST", &output);
    std::vector<float> expect_data{1, 2, 5, 6};
    for (int i = 0; i < 4; ++i) {
      if (output.data<float>()[i] != expect_data[i]) {
        DLOG << "output[" << i << "]: " << output.data<float>()[i]
             << " != expect[" << i << "]: " << expect_data[i];
        return 1;
      }
    }
  }
  // case 4
  DLOG << "running first case 4";
  {
    std::vector<float> data{1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                            11, 12, 13, 14, 15, 16, 17, 18, 19, 20};
    input_x.Resize(framework::make_ddim({4, 5}));
    float *in_data = input_x.mutable_data<float>();
    for (int i = 0; i < data.size(); ++i) in_data[i] = data[i];
    input_x.set_lod({{0, 2, 4}});

    TestSequencePoolOp(input_x, "FIRST", &output);
    std::vector<float> expect_data{1, 2, 3, 4, 5, 11, 12, 13, 14, 15};
    for (int i = 0; i < 10; ++i) {
      if (output.data<float>()[i] != expect_data[i]) {
        DLOG << "output[" << i << "]: " << output.data<float>()[i]
             << " != expect[" << i << "]: " << expect_data[i];
        return 1;
      }
    }
  }
  return 0;
}
