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

#include <cmath>
#include <iostream>
#include "../test_include.h"
#include "operators/compare_op.h"

namespace paddle_mobile {

template <typename T>
void LessThan(const framework::Tensor *X, const framework::Tensor *Y,
              const int Axis, framework::Tensor *Out) {
  const T *x = X->data<T>();
  const T *y = Y->data<T>();
  bool *output = Out->mutable_data<bool>();
  const auto &x_dims = X->dims();
  const auto &y_dims = Y->dims();
  /// axis = -1 represent the last dimensions.
  int axis = (Axis == -1 ? x_dims.size() - y_dims.size() : Axis);
  int batch = 1;
  int channels = 1;
  int elementwise_num = 1;
  for (int i = 0; i < axis; ++i) {
    batch *= x_dims[i];
  }
  for (int i = 0; i < y_dims.size(); ++i) {
    channels *= y_dims[i];
  }
  for (int i = y_dims.size() + axis; i < x_dims.size(); ++i) {
    elementwise_num *= x_dims[i];
  }
  // less than
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < channels; ++j) {
      int x_offset = (i * channels + j) * elementwise_num;
      int y_offset = j * elementwise_num;
      for (int k = 0; k < elementwise_num; ++k) {
        output[x_offset + k] = (x[x_offset + k] < y[y_offset]);
      }
    }
  }
}

template <typename T>
int TestLessThanOp(const std::vector<int> &x_shape,
                   const std::vector<int> &y_shape, const int axis) {
  framework::DDim xdims = framework::make_ddim(x_shape);
  framework::DDim ydims = framework::make_ddim(y_shape);
  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["X"] = std::vector<std::string>({"inputx"});
  inputs["Y"] = std::vector<std::string>({"inputy"});
  outputs["Out"] = std::vector<std::string>({"output"});

  auto inputx_var = scope.get()->Var("inputx");
  auto inputx = inputx_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<T>(inputx, xdims, static_cast<T>(-100), static_cast<T>(100));
  auto inputy_var = scope.get()->Var("inputy");
  auto inputy = inputy_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<T>(inputy, ydims, static_cast<T>(-100), static_cast<T>(100));

  auto output_var = scope.get()->Var("output");

  framework::AttributeMap attrs;
  attrs["axis"].Set<int>(axis);
  auto *op = new operators::LessThanOp<CPU, float>("less_than", inputs, outputs,
                                                   attrs, scope.get());
  op->InferShape();
  op->Init();
  op->Run();

  auto output = output_var->template Get<framework::LoDTensor>();

  framework::Tensor output_cmp;
  bool *output_cmp_data = output_cmp.mutable_data<bool>(output->dims());
  LessThan<T>(inputx, inputy, axis, &output_cmp);

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
  delete op;
  return 0;
}

}  // namespace paddle_mobile

int main() {
  paddle_mobile::TestLessThanOp<float>({1, 2, 3}, {1, 2, 3}, 0);
  paddle_mobile::TestLessThanOp<float>({10, 2, 1}, {10, 2, 1}, 0);

  paddle_mobile::TestLessThanOp<float>({2, 10, 1}, {1, 10, 1}, 1);
  paddle_mobile::TestLessThanOp<float>({10, 2, 1}, {1, 2, 1}, 1);

  paddle_mobile::TestLessThanOp<int64_t>({1, 2, 3}, {1, 2, 3}, 0);
  paddle_mobile::TestLessThanOp<int64_t>({10, 2, 1}, {10, 2, 1}, 0);

  paddle_mobile::TestLessThanOp<int64_t>({2, 10, 1}, {1, 10, 1}, 1);
  paddle_mobile::TestLessThanOp<int64_t>({10, 2, 1}, {1, 2, 1}, 1);

  std::cout << "test less_than op pass." << std::endl;
  return 0;
}
