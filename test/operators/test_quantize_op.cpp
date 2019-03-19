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
#include "../test_helper.h"
#include "../test_include.h"
#include "operators/quantize_op.h"

namespace paddle_mobile {
namespace round {
enum RoundType {
  RoundToEven = 0,
  RoundAwayZero = 1,
  RoundTowardsZero = 2,
};
}

template <round::RoundType T>
struct Round {
  int8_t operator()(float x);
};

template <>
struct Round<round::RoundAwayZero> {
  int8_t operator()(float x) { return std::round(x); }
};

template <>
struct Round<round::RoundTowardsZero> {
  int8_t operator()(float x) { return int8_t(x); }
};

template <>
struct Round<round::RoundToEven> {
  int8_t operator()(float x) {
    float v = std::round(x);
    int32_t q = static_cast<int32_t>(v);
    if (abs(abs(q - v) - 0.5) <= 0) {
      if (abs(q) % 2 != 0) {
        q = q + ((q > 0) ? -1 : 1);
      }
    }
    return static_cast<int8_t>(q);
  }
};

template <round::RoundType T>
static void quantize(const Tensor *input, const float scale, Tensor *output) {
  int batch_size = input->dims()[0];
  int channels = input->dims()[1];
  int input_h = input->dims()[2];
  int input_w = input->dims()[3];
  int output_h = output->dims()[2];
  int output_w = output->dims()[3];
  size_t input_spatial = input_h * input_w;
  size_t output_spatial = output_h * output_w;
  const float *x = input->data<const float>();
  int8_t *y = output->mutable_data<int8_t>();

  for (int nc = 0; nc < batch_size * channels; ++nc) {
    const float *xh = x + nc * input_spatial;
    int8_t *yh = y + nc * output_spatial;
    for (int h = 0; h < input_h; ++h, yh += output_w, xh += input_w) {
      for (int w = 0; w < input_w; ++w) {
        yh[w] = Round<T>()(xh[w] * scale);
      }
    }
  }
}

static float find_abs_max(const Tensor *input) {
  float max_abs = 0.f;
  const float *x = input->data<const float>();
  size_t size = input->numel();
  for (size_t i = 0; i < size; ++i) {
    float value = std::abs(x[i]);
    if (value > max_abs) {
      max_abs = value;
    }
  }
  return max_abs;
}

int TestQuqntizeOp(const int batch_size, const int channel, const int height,
                   const int width) {
  DLOG << "batch_size: " << batch_size << ", channel: " << channel
       << ", height: " << height << ", width: " << width;
  framework::DDim dim =
      framework::make_ddim({batch_size, channel, height, width});

  VariableNameMap inputs;
  VariableNameMap outputs;
  auto scope = std::make_shared<framework::Scope>();
  inputs["X"] = std::vector<std::string>({"input"});
  outputs["Out"] = std::vector<std::string>({"output"});
  outputs["OutScale"] = std::vector<std::string>({"output_scale"});

  auto input_var = scope.get()->Var("input");
  auto input = input_var->template GetMutable<framework::LoDTensor>();
  SetupTensor<float>(input, dim, -100.f, 100.f);

  auto output_var = scope.get()->Var("output");
  auto output_scale_var = scope.get()->Var("output_scale");

  framework::AttributeMap attrs;
  auto *op = new operators::QuantizeOp<CPU, float>("quantize", inputs, outputs,
                                                   attrs, scope.get());
  op->InferShape();
  op->Run();

  auto output = output_var->template Get<framework::LoDTensor>();
  const int8_t *output_data = output->data<int8_t>();
  auto output_scale = output_scale_var->template Get<framework::LoDTensor>();
  const float *output_scale_data = output_scale->data<float>();

  float output_scale_cmp = find_abs_max(input);
  PADDLE_MOBILE_ENFORCE(output_scale_cmp == output_scale_data[0],
                        "output_scale = %.6f, output_scale_cmp = %.6f",
                        output_scale_cmp, output_scale_data[0]);

  framework::Tensor output_cmp;
  output_cmp.Resize(output->dims());
  float scale = 127 / output_scale_cmp;
  quantize<round::RoundTowardsZero>(input, scale, &output_cmp);
  int8_t *output_cmp_data = output_cmp.data<int8_t>();
  for (int i = 0; i < output->numel(); ++i) {
    PADDLE_MOBILE_ENFORCE(output_data[i] == output_cmp_data[i],
                          "output[%d] = %d, output_cmp[%d] = %d", i,
                          static_cast<int>(output_data[i]), i,
                          static_cast<int>(output_cmp_data[i]));
  }
  delete op;
  return 0;
}

}  // namespace paddle_mobile

int main(int argc, char *argv[]) {
  TestQuqntizeOp(1, 10, 10, 5);
  TestQuqntizeOp(1, 111, 111, 5);
  TestQuqntizeOp(5, 111, 111, 5);
}
