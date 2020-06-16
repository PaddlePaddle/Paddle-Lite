// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <iostream>
#include <memory>
#include <vector>

#include "compute_api.h"    // NOLINT
#include "compute_param.h"  // NOLINT
#include "compute_utils.h"  // NOLINT
#include "paddle_api.h"     // NOLINT
#include "utils.h"          // NOLINT

using namespace paddle::lite_api;  // NOLINT

void activation_naive_impl(const float* din,
                           float* dout,
                           int64_t len,
                           ActivationType act_type,
                           float leaky_relu_alpha) {
  switch (act_type) {
    case ActivationType::kRelu: {
      for (int i = 0; i < len; i++) {
        dout[i] = std::max(0.f, din[i]);
      }
      break;
    }
    case ActivationType::kRelu6: {
      for (int i = 0; i < len; i++) {
        dout[i] = std::max(0.f, din[i]);
        dout[i] = std::min(6.f, dout[i]);
      }
      break;
    }
    case ActivationType::kLeakyRelu: {
      for (int i = 0; i < len; i++) {
        dout[i] = din[i] > 0.f ? din[i] : din[i] * leaky_relu_alpha;
      }
      break;
    }
    case ActivationType::kSigmoid: {
      for (int i = 0; i < len; i++) {
        dout[i] = 1.f / (1.f + std::exp(-din[i]));
      }
      break;
    }
    case ActivationType::kTanh: {
      for (int i = 0; i < len; i++) {
        dout[i] = (std::exp(din[i]) - std::exp(-din[i])) /
                  (std::exp(din[i]) + std::exp(-din[i]));
      }
      break;
    }
    default:
      std::cerr << "the type of activation is unknow." << std::endl;
      assert(0);
  }
}

void activation_func(int n,
                     int c,
                     int h,
                     int w,
                     ActivationType act_type,
                     float leaky_relu_alpha,
                     int warmup,
                     int repeats,
                     bool check_result,
                     int threads,
                     PowerMode power_mode) {
  Tensor input, output, output_ref;
  input.Resize({n, c, h, w});
  input.set_precision(PRECISION(kFloat));
  output_ref.Resize({n, c, h, w});
  output_ref.set_precision(PRECISION(kFloat));
  fill_tensor_rand(input, -1.f, 1.f);
  ComputeEngine<TARGET(kARM)>::env_init(power_mode, threads);
  ComputeEngine<TARGET(kARM)> act;

  ActivationParam act_param;
  act_param.active_type = act_type;
  act_param.X = &input;
  act_param.Out = &output;
  act_param.Leaky_relu_alpha = leaky_relu_alpha;
  std::string act_str;
  if (act_type == ActivationType::kRelu) {
    act_str = "relu";
  } else if (act_type == ActivationType::kRelu6) {
    act_str = "relu6";
  } else if (act_type == ActivationType::kLeakyRelu) {
    act_str = "leaky_relu";
  } else if (act_type == ActivationType::kSigmoid) {
    act_str = "sigmoid";
  } else if (act_type == ActivationType::kTanh) {
    act_str = "tanh";
  } else {
    std::cerr << "act type: " << static_cast<int>(act_type)
              << "is not support now." << std::endl;
    assert(0);
  }
  act.CreateOperator(act_str.c_str());
  act.SetParam(&act_param);
  act.Launch();
  if (output.shape() != output_ref.shape()) {
    std::cerr << "act op infer shape error." << std::endl;
    assert(0);
  }
  Timer t;
  for (int i = 0; i < warmup; ++i) {
    act.Launch();
  }

  for (int i = 0; i < repeats; ++i) {
    t.Start();
    act.Launch();
    t.Stop();
  }
  auto shape = input.shape();
  std::cout << "act input shape: " << shape[0] << ", " << shape[1] << ", "
            << shape[2] << ", " << shape[3]
            << ", act_type: " << static_cast<int>(act_type)
            << ", warmup: " << warmup << ", repeats: " << repeats
            << ", power mode: " << static_cast<int>(power_mode)
            << ", threads: " << threads << ", avg time: " << t.LapTimes().Avg()
            << " ms" << std::endl;

  if (check_result) {
    const float* din = input.data<float>();
    float* dout_ref = output_ref.mutable_data<float>();
    int64_t len = dim_production(input);
    activation_naive_impl(din, dout_ref, len, act_type, leaky_relu_alpha);
    double max_ratio = 0;
    double max_diff = 0;
    tensor_cmp_host(output, output_ref, max_ratio, max_diff);
    if (std::abs(max_ratio) > 1e-3f) {
      if (max_diff > 5e-4f) {
        std::cout << "basic result" << std::endl;
        print_tensor(output_ref);
        std::cout << "lite result" << std::endl;
        print_tensor(output);
        Tensor tdiff;
        tdiff.set_precision(PRECISION(kFloat));
        tensor_diff(output_ref, output, tdiff);
        std::cout << "diff result" << std::endl;
        print_tensor(tdiff);
        tdiff.ReleaseRawTensor();
      }
    }
  }

  input.ReleaseRawTensor();
  output.ReleaseRawTensor();
  output_ref.ReleaseRawTensor();
}

static int basic_test = 1;
static int n = 1;
static int c = 3;
static int h = 224;
static int w = 224;
static int act_type = 1;
static float leaky_relu_alpha = 2.f;
static int warmup = 0;
static int repeats = 1;
static int check_result = 1;
static int power_mode = 3;
static int threads = 1;

int main(int argc, const char** argv) {
  if (argc < 2) {
    std::cout << "usage: ./" << argv[0]
              << "basic_test n c h w act_type leaky_relu_alpha"
                 " warmup repeats check_result power_mode threads"
              << std::endl;
    return 0;
  }
  if (argc >= 2) {
    basic_test = atoi(argv[1]) > 0;
  }
  if (argc >= 3) {
    n = atoi(argv[2]);
  }
  if (argc >= 4) {
    c = atoi(argv[3]);
  }
  if (argc >= 5) {
    h = atoi(argv[4]);
  }
  if (argc >= 6) {
    w = atoi(argv[5]);
  }
  if (argc >= 7) {
    act_type = atoi(argv[6]);
  }
  if (argc >= 8) {
    leaky_relu_alpha = atof(argv[7]);
  }
  if (argc >= 9) {
    warmup = atoi(argv[8]);
  }
  if (argc >= 10) {
    repeats = atoi(argv[9]);
  }
  if (argc >= 11) {
    check_result = atoi(argv[10]);
  }
  if (argc >= 12) {
    power_mode = atoi(argv[11]);
  }
  if (argc >= 13) {
    threads = atoi(argv[12]);
  }
  // basic test
  if (basic_test) {
    std::cout << "RUN BASIC TEST BEGIN: " << std::endl;
    for (auto& n : {1, 3, 4}) {
      for (auto& c : {1, 3, 32}) {
        for (auto& h : {5, 64, 112, 224}) {
          for (auto& w : {5, 64, 112, 224}) {
            for (auto& act_type : {1, 2, 4, 5, 6}) {
              for (auto& threads : {1, 2, 4}) {
                activation_func(n,
                                c,
                                h,
                                w,
                                static_cast<ActivationType>(act_type),
                                leaky_relu_alpha,
                                0,
                                1,
                                1,
                                threads,
                                static_cast<PowerMode>(3));
              }
            }
          }
        }
      }
    }
    std::cout << "RUN BASIC TEST END: " << std::endl;
  }

  // costum test
  std::cout << "RUN CUSTOM TEST BEGIN: " << std::endl;
  activation_func(n,
                  c,
                  h,
                  w,
                  static_cast<ActivationType>(act_type),
                  leaky_relu_alpha,
                  warmup,
                  repeats,
                  check_result,
                  threads,
                  static_cast<PowerMode>(power_mode));
  std::cout << "RUN CUSTOM TEST END: " << std::endl;
  return 0;
}
