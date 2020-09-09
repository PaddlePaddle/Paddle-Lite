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

#include "lite/backends/cuda/math/gru_forward.h"
#include "lite/core/device_info.h"

namespace paddle {
namespace lite {
namespace cuda {
namespace math {

/*
 * threads(frame_per_block, batch_per_block)
 * grid(frame_blocks, batch_blocks)
 */
template <typename T>
__global__ void GruForwardResetOutput(
    T* gate_value,
    T* reset_output_value,
    T* prev_output_value,
    int frame_size,
    int batch_size,
    lite::cuda::math::ActivationType active_gate,
    bool is_batch) {
  const int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame_idx >= frame_size) return;

  int batch_idx = 0;
  if (is_batch) {
    batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size) return;
    gate_value += batch_idx * 3 * frame_size;
    reset_output_value += batch_idx * frame_size;
  }
  T prev_out = 0;
  T reset_out_val;
  T update_gate_value = gate_value[frame_idx + frame_size * 0];
  T reset_gate_value = gate_value[frame_idx + frame_size * 1];

  if (prev_output_value) {
    if (is_batch) {
      prev_output_value += batch_idx * frame_size;
    }
    prev_out = prev_output_value[frame_idx];
  }

  if (active_gate == lite::cuda::math::ActivationType::kSigmoid) {
    update_gate_value = Sigmoid(update_gate_value);
    reset_gate_value = Sigmoid(reset_gate_value);
  } else if (active_gate == lite::cuda::math::ActivationType::kReLU) {
    update_gate_value = ReLU(update_gate_value);
    reset_gate_value = ReLU(reset_gate_value);
  } else if (active_gate == lite::cuda::math::ActivationType::kTanh) {
    update_gate_value = Tanh(update_gate_value);
    reset_gate_value = Tanh(reset_gate_value);
  }

  reset_out_val = prev_out * reset_gate_value;

  gate_value[frame_idx + frame_size * 0] = update_gate_value;
  gate_value[frame_idx + frame_size * 1] = reset_gate_value;
  reset_output_value[frame_idx] = reset_out_val;
}

template <>
__global__ void GruForwardResetOutput(
    half* gate_value,
    half* reset_output_value,
    half* prev_output_value,
    int frame_size,
    int batch_size,
    lite::cuda::math::ActivationType active_gate,
    bool is_batch) {
  const int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame_idx >= frame_size) return;

  int batch_idx = 0;
  if (is_batch) {
    batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size) return;
    gate_value += batch_idx * 3 * frame_size;
    reset_output_value += batch_idx * frame_size;
  }
  half prev_out = 0;
  half reset_out_val;
  half update_gate_value = gate_value[frame_idx + frame_size * 0];
  half reset_gate_value = gate_value[frame_idx + frame_size * 1];

  if (prev_output_value) {
    if (is_batch) {
      prev_output_value += batch_idx * frame_size;
    }
    prev_out = prev_output_value[frame_idx];
  }

  if (active_gate == ActivationType::kSigmoid) {
    update_gate_value = Sigmoid(update_gate_value);
    reset_gate_value = Sigmoid(reset_gate_value);
  } else if (active_gate == ActivationType::kReLU) {
    update_gate_value = ReLU(update_gate_value);
    reset_gate_value = ReLU(reset_gate_value);
  } else if (active_gate == ActivationType::kTanh) {
    update_gate_value = Tanh(update_gate_value);
    reset_gate_value = Tanh(reset_gate_value);
  }
#if __CUDA_ARCH__ >= 530
  reset_out_val = __hmul(prev_out, reset_gate_value);
#else
  reset_out_val =
      __float2half(__half2float(prev_out) * __half2float(reset_gate_value));
#endif

  gate_value[frame_idx + frame_size * 0] = update_gate_value;
  gate_value[frame_idx + frame_size * 1] = reset_gate_value;
  reset_output_value[frame_idx] = reset_out_val;
}

/*
 * threads(frame_per_block, batch_per_block)
 * grid(frame_blocks, batch_blocks)
 */
template <typename T>
__global__ void GruForwardFinalOutput(
    T* gate_value,
    T* prev_output_value,
    T* output_value,
    int frame_size,
    int batch_size,
    lite::cuda::math::ActivationType active_node,
    bool origin_mode,
    bool is_batch) {
  const int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame_idx >= frame_size) return;
  int batch_idx = 0;
  if (is_batch) {
    batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size) {
      return;
    }
    gate_value += batch_idx * 3 * frame_size;
    output_value += batch_idx * frame_size;
  }

  T output;
  T prev_out = 0;
  T update_gate_value = gate_value[frame_idx + frame_size * 0];
  T state_frame_value = gate_value[frame_idx + frame_size * 2];

  if (prev_output_value) {
    if (is_batch) prev_output_value += batch_idx * frame_size;
    prev_out = prev_output_value[frame_idx];
  }

  if (active_node == lite::cuda::math::ActivationType::kSigmoid) {
    state_frame_value = Sigmoid(state_frame_value);
  } else if (active_node == lite::cuda::math::ActivationType::kReLU) {
    state_frame_value = ReLU(state_frame_value);
  } else if (active_node == lite::cuda::math::ActivationType::kTanh) {
    state_frame_value = Tanh(state_frame_value);
  }

  if (origin_mode) {
    output = update_gate_value * prev_out + state_frame_value -
             update_gate_value * state_frame_value;
  } else {
    output = prev_out - update_gate_value * prev_out +
             update_gate_value * state_frame_value;
  }

  gate_value[frame_idx + frame_size * 2] = state_frame_value;
  output_value[frame_idx] = output;
}

template <>
__global__ void GruForwardFinalOutput(
    half* gate_value,
    half* prev_output_value,
    half* output_value,
    int frame_size,
    int batch_size,
    lite::cuda::math::ActivationType active_node,
    bool origin_mode,
    bool is_batch) {
  const int frame_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame_idx >= frame_size) return;
  int batch_idx = 0;
  if (is_batch) {
    batch_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (batch_idx >= batch_size) {
      return;
    }
    gate_value += batch_idx * 3 * frame_size;
    output_value += batch_idx * frame_size;
  }

  half output;
  half prev_out = 0;
  half update_gate_value = gate_value[frame_idx + frame_size * 0];
  half state_frame_value = gate_value[frame_idx + frame_size * 2];

  if (prev_output_value) {
    if (is_batch) prev_output_value += batch_idx * frame_size;
    prev_out = prev_output_value[frame_idx];
  }

  if (active_node == lite::cuda::math::ActivationType::kSigmoid) {
    state_frame_value = Sigmoid(state_frame_value);
  } else if (active_node == lite::cuda::math::ActivationType::kReLU) {
    state_frame_value = ReLU(state_frame_value);
  } else if (active_node == lite::cuda::math::ActivationType::kTanh) {
    state_frame_value = Tanh(state_frame_value);
  }

  if (origin_mode) {
#if __CUDA_ARCH__ >= 530
    output =
        __hsub(__hadd(__hmul(update_gate_value, prev_out), state_frame_value),
               __hmul(update_gate_value, state_frame_value));
#else
    output = __float2half(
        __half2float(update_gate_value) * __half2float(prev_out) +
        __half2float(state_frame_value) -
        __half2float(update_gate_value) * __half2float(state_frame_value));
#endif
  } else {
#if __CUDA_ARCH__ >= 530
    output = prev_out - update_gate_value * prev_out +
             update_gate_value * state_frame_value;
    output = __hadd(__hsub(prev_out, __hmul(update_gate_value, prev_out)),
                    __hmul(update_gate_value, state_frame_value));
#else
    output = __float2half(
        __half2float(prev_out) -
        __half2float(update_gate_value) * __half2float(prev_out) +
        __half2float(update_gate_value) * __half2float(state_frame_value));
#endif
  }

  gate_value[frame_idx + frame_size * 2] = state_frame_value;
  output_value[frame_idx] = output;
}

template __global__ void GruForwardFinalOutput<float>(
    float* gate_value,
    float* prev_output_value,
    float* output_value,
    int frame_size,
    int batch_size,
    lite::cuda::math::ActivationType active_node,
    bool origin_mode,
    bool is_batch);

template __global__ void GruForwardResetOutput<float>(
    float* gate_value,
    float* reset_output_value,
    float* prev_output_value,
    int frame_size,
    int batch_size,
    lite::cuda::math::ActivationType active_gate,
    bool is_batch);

}  // namespace math
}  // namespace cuda
}  // namespace lite
}  // namespace paddle
