#pragma once
#include "common/log.h"
#include "framework/ddim.h"
#include "framework/tensor.h"
#include <random>

template <typename T>
void SetupTensor(paddle_mobile::framework::Tensor *input,
                 paddle_mobile::framework::DDim dims, T lower, T upper) {
  static unsigned int seed = 100;
  std::mt19937 rng(seed++);
  std::uniform_real_distribution<double> uniform_dist(0, 1);

  T *input_ptr = input->mutable_data<T>(dims);
  for (int i = 0; i < input->numel(); ++i) {
    input_ptr[i] = static_cast<T>(uniform_dist(rng) * (upper - lower) + lower);
  }
}
