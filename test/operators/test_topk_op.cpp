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
#include "operators/top_k_op.h"

namespace paddle_mobile {

void TopK(const framework::Tensor *X, framework::Tensor *Y,
          framework::Tensor *Indices, const int K) {
  const float *x = X->data<float>();
  float *y = Y->mutable_data<float>();
  int64_t *indices = Indices->mutable_data<int64_t>();

  int dim_size = X->dims().size();
  int row = 1;
  int col = X->dims()[dim_size - 1];
  for (int i = 0; i < dim_size - 1; ++i) {
    row *= X->dims()[i];
  }

  std::vector<float> vec(col);
  for (int i = 0; i < row; ++i) {
    for (int j = 0; j < col; ++j) {
      vec[j] = x[i * col + j];
    }
    for (int k = 0; k < K; ++k) {
      float max = vec[0];
      int index = 0;
      for (int j = 1; j < col; ++j) {
        if (vec[j] > max) {
          max = vec[j];
          index = j;
        }
      }
      y[i * K + k] = max;
      indices[i * K + k] = index;
      vec[index] = -std::numeric_limits<float>::max();
    }
  }
}

int TestTopKOp(const std::vector<int> input_shape, const int K) {
  framework::DDim dims = framework::make_ddim(input_shape);
  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["X"] = std::vector<std::string>({"input"});
  outputs["Out"] = std::vector<std::string>({"output"});
  outputs["Indices"] = std::vector<std::string>({"indices"});

  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(input, dims, -100.0, 100.0);

  auto output_var = scope.get()->Var("output");
  auto indices_var = scope.get()->Var("indices");

  framework::AttributeMap attrs;
  attrs["k"].Set<int>(K);
  auto *op = new operators::TopKOp<CPU, float>("top_k", inputs, outputs, attrs,
                                               scope.get());
  op->InferShape();
  op->Init();
  op->Run();

  auto output = output_var->template Get<framework::LoDTensor>();
  auto indices = indices_var->template Get<framework::LoDTensor>();

  framework::Tensor output_cmp, indices_cmp;
  float *output_cmp_data = output_cmp.mutable_data<float>(output->dims());
  int64_t *indices_cmp_data =
      indices_cmp.mutable_data<int64_t>(indices->dims());
  TopK(input, &output_cmp, &indices_cmp, K);

  // sort output
  float *output_data = const_cast<float *>(output->data<float>());
  int64_t *indices_data = const_cast<int64_t *>(indices->data<int64_t>());
  //  std::vector<std::pair<float, size_t>> vec(K);
  //  for (int i = 0; i < output->numel() / K; ++i) {
  //    for (int j = 0; j < K; ++j) {
  //      vec[j] = std::move(std::make_pair(output_data[i * K + j],
  //      indices_data[i * K + j]));
  //    }
  //    std::sort(vec.begin(), vec.end(),
  //              [](const std::pair<float, size_t> &l,
  //                 const std::pair<float, size_t> &r) {
  //                   return l.first > r.first; });
  //    for (int j = 0; j < K; ++j) {
  //      output_data[i * K + j] = vec[j].first;
  //      indices_data[i * K + j] = vec[j].second;
  //    }
  //  }

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

  for (int i = 0; i < indices->numel(); ++i) {
    if (indices_data[i] != indices_cmp_data[i]) {
      LOG(kLOG_INFO) << "indices_data[" << i << "] = " << indices_data[i]
                     << ", indices_cmp_data[" << i
                     << "] = " << indices_cmp_data[i];
      delete op;
      exit(1);
    }
  }
  delete op;
  return 0;
}

}  // namespace paddle_mobile

int main(int argc, char *argv[]) {
  TestTopKOp({1, 100}, 1);
  TestTopKOp({128, 100}, 10);
  TestTopKOp({128, 2, 100}, 10);
  return 0;
}
