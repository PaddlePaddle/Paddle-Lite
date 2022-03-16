// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <limits>
#include "driver/nvidia_tensorrt/converter/plugin/arg_max.h"

namespace nnadapter {
namespace nvidia_tensorrt {

ArgMaxPluginDynamic::ArgMaxPluginDynamic() {}

ArgMaxPluginDynamic::ArgMaxPluginDynamic(int axis) : _axis(axis) {}

ArgMaxPluginDynamic::ArgMaxPluginDynamic(const void* serial_data,
                                         size_t serial_length) {
  Deserialize(&serial_data, &serial_length, &_axis);
}

nvinfer1::IPluginV2DynamicExt* ArgMaxPluginDynamic::clone() const noexcept {
  return new ArgMaxPluginDynamic(_axis);
}

template <typename InType, typename OutType, unsigned TPB>
__global__ void arg_max_kernel(const InType* input,
                               OutType* output,
                               int pre,
                               int axis_num,
                               int post,
                               const InType init) {
  int height = pre * post;
  int width = axis_num;
  int post_size = post;
  __shared__ int block_pair_idx[TPB];
  __shared__ InType block_pair_val[TPB];

  for (int idx = blockIdx.x; idx < height; idx += gridDim.x) {
    int kv_pair_idx = -1;
    InType kv_pair_val = init;
    int h = idx / post_size;
    int w = idx % post_size;

    for (int k = threadIdx.x; k < width; k += blockDim.x) {
      int index = h * width * post_size + k * post_size + w;
      if (input[index] > kv_pair_val) {
        kv_pair_val = input[index];
        kv_pair_idx = k;
      }
    }
    block_pair_idx[threadIdx.x] = kv_pair_idx;
    block_pair_val[threadIdx.x] = kv_pair_val;
    __syncthreads();

    if (0 == threadIdx.x) {
      int kv_pair_idx_1 = -1;
      InType kv_pair_val_1 = init;
      for (int i = 0; i < TPB; i++) {
        if (block_pair_val[i] > kv_pair_val_1) {
          kv_pair_idx_1 = block_pair_idx[i];
          kv_pair_val_1 = block_pair_val[i];
        }
      }
      output[idx] = static_cast<OutType>(kv_pair_idx_1);
    }
    __syncthreads();
  }
}

int32_t ArgMaxPluginDynamic::enqueue(
    const nvinfer1::PluginTensorDesc* input_desc,
    const nvinfer1::PluginTensorDesc* output_desc,
    const void* const* inputs,
    void* const* outputs,
    void* workspace,
    cudaStream_t stream) noexcept {
  auto input_dims = input_desc[0].dims;
  auto axis_num = input_dims.d[_axis];
  int pre = 1;
  int post = 1;
  for (int i = 0; i < _axis; i++) {
    pre *= input_dims.d[i];
  }
  for (int i = _axis + 1; i < input_dims.nbDims; i++) {
    post *= input_dims.d[i];
  }
  const int block_size = 128;
  const int grid_size = (axis_num + block_size - 1) / block_size;

  if (input_desc[0].type == nvinfer1::DataType::kFLOAT) {
    const float* input = static_cast<const float*>(inputs[0]);
    int* output = static_cast<int*>(outputs[0]);
    auto init = std::numeric_limits<float>::lowest();
    arg_max_kernel<float,
                   int,
                   block_size><<<grid_size, block_size, 0, stream>>>(
        input, output, pre, axis_num, post, init);
  } else {
    NNADAPTER_LOG(FATAL)
        << "ArgMax only support float-input and int-output for now.";
  }
  return 0;
}

size_t ArgMaxPluginDynamic::getSerializationSize() const noexcept {
  return SerializedSize(_axis);
}

void ArgMaxPluginDynamic::serialize(void* buffer) const noexcept {
  Serialize(&buffer, _axis);
}

REGISTER_NNADAPTER_TENSORRT_PLUGIN(ArgMaxPluginDynamic,
                                   ArgMaxPluginDynamicCreator,
                                   "arg_max_plugin_dynamic");

}  // namespace nvidia_tensorrt
}  // namespace nnadapter
