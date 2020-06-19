/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>

#include "lite/backends/fpga/KD/pe.hpp"
#include "lite/backends/fpga/KD/pe_params.hpp"
namespace paddle {
namespace zynqmp {

class SplitPE : public PE {
 public:
  bool init() {
    std::vector<Tensor*> outputs = param_.outputs;
    for (size_t i = 0; i < outputs.size(); i++) {
      Tensor* out = outputs[i];
      out->setAligned(false);
      out->setDataLocation(CPU);
    }
    return true;
  }

  std::vector<int> stride_numel(std::vector<int> ddim) {
    std::vector<int> strides(ddim.size());
    strides[ddim.size() - 1] = ddim[ddim.size() - 1];
    for (int i = ddim.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * ddim[i];
    }
    return strides;
  }

  template <typename T>
  inline void StridedNumelCopyWithAxis(int64_t axis,
                                       T* dst,
                                       const std::vector<int>& dst_stride_numel,
                                       T* src,
                                       const std::vector<int>& src_stride_numel,
                                       int64_t size) {
    int64_t before = dst_stride_numel[0] / dst_stride_numel[axis];
    int64_t src_after = src_stride_numel[axis];
    int64_t dst_after = dst_stride_numel[axis];

    // PADDLE_MOBILE_ENFORCE(src_stride_numel.size() == dst_stride_numel.size(),
    //                       "src and dst tensor should have the same dims
    //                       size.");

    for (int64_t i = 0; i < axis; ++i) {
      if (i < axis) {
        // PADDLE_MOBILE_ENFORCE(src_stride_numel[i] / src_stride_numel[axis] ==
        //                           dst_stride_numel[i] /
        //                           dst_stride_numel[axis],
        //                       "src and dst should have the same elements "
        //                       "except the specified axis.");
      } else if (i == axis) {
        continue;
      } else {
        // PADDLE_MOBILE_ENFORCE(src_stride_numel[i] == dst_stride_numel[i],
        //                       "src and dst should have the same elements "
        //                       "except the specified axis.");
      }
    }

    for (int64_t i = 0; i < before; ++i) {
      memcpy(dst + i * dst_after, src + i * src_after, sizeof(T) * size);
    }
  }

  void split3D() {
    int axis = param_.axis;
    // float16* dst = param_.output->data<float16>();
    // std::vector<int>& dst_dims = ;
    // StridedNumelCopyWithAxis();
  }

  bool dispatch() {
    Tensor* input = param_.input;
    input->syncToCPU();
    if (input->shape().dimSize() <= 3) {
      auto in_stride = stride_numel(input->shape().dims());
      int64_t axis = param_.axis;
      size_t input_offset = 0;
      float16* in_data = input->data<float16>();

      for (auto& out : param_.outputs) {
        float16* out_data = out->mutableData<float16>();
        auto out_stride = stride_numel(out->shape().dims());

        StridedNumelCopyWithAxis<float16>(axis,
                                          out_data,
                                          out_stride,
                                          in_data + input_offset,
                                          in_stride,
                                          out_stride[axis]);
        input_offset += out_stride[axis];
        out->flush();
      }
      return true;
    }

    std::vector<Tensor*> outputs = param_.outputs;

    int in_channel = input->shape().channel();
    // int split_channel = input->shape().channel() / param_.num;
    int hw = input->shape().height() * input->shape().width();

    float16* in_data = input->data<float16>();

    for (int i = 0; i < hw; i++) {
      int channel_stride = 0;
      for (int n = 0; n < outputs.size(); n++) {
        Tensor* out = outputs[n];
        float16* out_data = out->data<float16>();
        memcpy(out_data + i * out->shape().channel(),
               in_data + i * in_channel + channel_stride,
               out->shape().channel() * sizeof(float16));
        channel_stride += out->shape().channel();
      }
    }

    for (int n = 0; n < outputs.size(); n++) {
      Tensor* out = outputs[n];
      out->flush();
      out->copyScaleFrom(input);
    }
    return true;
  }

  SplitParam& param() { return param_; }

 private:
  SplitParam param_;
};

}  // namespace zynqmp
}  // namespace paddle
