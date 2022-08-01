// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/kernels/host/max_pool_with_index_compute.h"
#include <cmath>
#include <string>
#include <vector>
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace host {

inline int AdaptStartIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      floor(static_cast<double>(ph * input_size) / output_size));
}

inline int AdaptEndIndex(int ph, int input_size, int output_size) {
  return static_cast<int>(
      ceil(static_cast<double>((ph + 1) * input_size) / output_size));
}

#define COMMON_LOOP_BODY                           \
  float ele = static_cast<float>(-FLT_MAX);        \
  int index = -1;                                  \
  for (int h = hstart; h < hend; ++h) {            \
    for (int w = wstart; w < wend; ++w) {          \
      if (ele < input_data[h * input_width + w]) { \
        ele = input_data[h * input_width + w];     \
        index = h * input_width + w;               \
      }                                            \
    }                                              \
  }                                                \
  output_data[ph * output_width + pw] = ele;       \
  mask_data[ph * output_width + pw] = index;

void MaxPoolWithIndexCompute::Run() {
  auto param = Param<param_t>();
  if (param.ksize.size() != 2)
    LOG(FATAL) << "MaxPoolWithIndex op only supports 2D input for now.";

  auto input_dim = param.x->dims();
  auto output_dim = param.output->dims();
  const int batch_size = input_dim[0];
  const int input_height = input_dim[2];
  const int input_width = input_dim[3];
  const int output_channels = output_dim[1];
  const int output_height = output_dim[2];
  const int output_width = output_dim[3];
  const int ksize_height = param.ksize[0];
  const int ksize_width = param.ksize[1];
  const int stride_height = param.strides[0];
  const int stride_width = param.strides[1];
  const int padding_height = (*param.paddings)[0];
  const int padding_width = (*param.paddings)[1];
  const int input_stride = input_height * input_width;
  const int output_stride = output_height * output_width;
  const float* input_data = param.x->data<float>();
  float* output_data = param.output->mutable_data<float>();
  float* mask_data = param.mask->mutable_data<float>();
  int hstart, hend;
  int wstart, wend;
  auto adaptive = param.adaptive;
  if (adaptive) {
    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int ph = 0; ph < output_height; ++ph) {
          hstart = AdaptStartIndex(ph, input_height, output_height);
          hend = AdaptEndIndex(ph, input_height, output_height);
          for (int pw = 0; pw < output_width; ++pw) {
            wstart = AdaptStartIndex(pw, input_width, output_width);
            wend = AdaptEndIndex(pw, input_width, output_width);
            COMMON_LOOP_BODY
          }
        }
        // offset
        input_data += input_stride;
        output_data += output_stride;
        mask_data += output_stride;
      }
    }
  } else {
    for (int i = 0; i < batch_size; i++) {
      for (int c = 0; c < output_channels; ++c) {
        for (int ph = 0; ph < output_height; ++ph) {
          hstart = ph * stride_height - padding_height;
          hend = std::min(hstart + ksize_height, input_height);
          hstart = std::max(hstart, 0);
          for (int pw = 0; pw < output_width; ++pw) {
            wstart = pw * stride_width - padding_width;
            wend = std::min(wstart + ksize_width, input_width);
            wstart = std::max(wstart, 0);
            COMMON_LOOP_BODY
          }
        }
        // offset
        input_data += input_stride;
        output_data += output_stride;
        mask_data += output_stride;
      }
    }
  }
}

#undef COMMON_LOOP_BODY
}  // namespace host
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

#ifdef LITE_BUILD_EXTRA
REGISTER_LITE_KERNEL(max_pool2d_with_index,
                     kHost,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::host::MaxPoolWithIndexCompute,
                     fp32)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Mask", {LiteType::GetTensorTy(TARGET(kHost))})
    .Finalize();
#endif
