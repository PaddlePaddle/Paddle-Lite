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

#ifdef SPLIT_OP

#include "operators/kernel/split_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool SplitKernel<GPU_CL, float>::Init(SplitParam<GPU_CL>* param) {
  this->cl_helper_.AddKernel("fetch", "fetch_kernel.cl");
  this->cl_helper_.AddKernel("feed", "feed_kernel.cl");
  return true;
}

// Strided numel memory copy from src to dst by the specified axis
//
// For example, for a tensor dims [4, 20, 100], the strieded numel is
// [8000, 2000, 100]
//
// NOTE: The src and dst tensor should have the same elements
// except the specified axis.
template <typename T>
void StridedNumelCopyWithAxis(int64_t axis, T* dst,
                              const framework::DDim& dst_stride_numel,
                              const T* src,
                              const framework::DDim& src_stride_numel,
                              int64_t size) {
  int64_t before = dst_stride_numel[0] / dst_stride_numel[axis];
  int64_t src_after = src_stride_numel[axis];
  int64_t dst_after = dst_stride_numel[axis];

  PADDLE_MOBILE_ENFORCE(src_stride_numel.size() == dst_stride_numel.size(),
                        "src and dst tensor should have the same dims size.");

  for (int64_t i = 0; i < axis; ++i) {
    if (i < axis) {
      PADDLE_MOBILE_ENFORCE(src_stride_numel[i] / src_stride_numel[axis] ==
                                dst_stride_numel[i] / dst_stride_numel[axis],
                            "src and dst should have the same elements "
                            "except the specified axis.");
    } else if (i == axis) {
      continue;
    } else {
      PADDLE_MOBILE_ENFORCE(src_stride_numel[i] == dst_stride_numel[i],
                            "src and dst should have the same elements "
                            "except the specified axis.");
    }
  }

  for (int64_t i = 0; i < before; ++i) {
    memory::Copy(dst + i * dst_after, src + i * src_after, sizeof(T) * size);
  }
}

template <>
void SplitKernel<GPU_CL, float>::Compute(const SplitParam<GPU_CL>& param) {
  auto kernel0 = this->cl_helper_.KernelAt(0);
  auto kernel1 = this->cl_helper_.KernelAt(1);
  auto* input_image = param.InputX();
  auto in_stride = framework::stride_numel(input_image->dims());
  auto input_dims = input_image->dims();
  auto outs_images = param.Outs();
  int64_t axis = param.Axis();

  Tensor* input_tensor = new Tensor();
  input_tensor->Resize(input_image->dims());
  input_tensor->mutable_data<float>();

  framework::CLImageToTensor(input_image, input_tensor,
                             this->cl_helper_.CLContext(),
                             this->cl_helper_.CLCommandQueue(), kernel0);

  size_t input_offset = 0;
  for (auto out : outs_images) {
    auto out_stride = framework::stride_numel(out->dims());

    Tensor* temp_out = new Tensor();
    temp_out->Resize(out->dims());
    temp_out->mutable_data<float>();
    framework::CLImageToTensor(out, temp_out, this->cl_helper_.CLContext(),
                               this->cl_helper_.CLCommandQueue(), kernel0);
    StridedNumelCopyWithAxis<float>(axis, temp_out->data<float>(), out_stride,
                                    input_tensor->data<float>() + input_offset,
                                    in_stride, out_stride[axis]);
    input_offset += out_stride[axis];
    out->InitEmptyImage(this->cl_helper_.CLContext(),
                        this->cl_helper_.CLCommandQueue(), temp_out->dims());
    framework::TensorToCLImage(temp_out, out, this->cl_helper_.CLContext(),
                               this->cl_helper_.CLCommandQueue(), kernel1);
    outs_images.push_back(out);

    delete (temp_out);
  }
  delete (input_tensor);
}

template class SplitKernel<GPU_CL, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
