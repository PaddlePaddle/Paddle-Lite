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

#ifdef IM2SEQUENCE_OP

#include "operators/kernel/im2sequence_kernel.h"

namespace paddle_mobile {
namespace operators {

template <>
bool Im2SequenceKernel<CPU, float>::Init(Im2SequenceParam<CPU> *para) {
  return true;
}

inline int Im2SeqOutputSize(int input_size, int filter_size, int padding_0,
                            int padding_1, int stride) {
  const int output_size =
      (input_size + padding_0 + padding_1 - filter_size) / stride + 1;
  return output_size;
}

template <>
void Im2SequenceKernel<CPU, float>::Compute(
    const Im2SequenceParam<CPU> &param) {
  const Tensor *in_x = param.Input();
  framework::LoDTensor *out = param.Output();
  out->mutable_data<float>();

  std::vector<int> kernels = param.Kernels();
  std::vector<int> strides = param.Strides();
  std::vector<int> paddings = param.Paddings();

  auto in_x_dim = in_x->dims();
  const int batch_size = static_cast<int>(in_x_dim[0]);
  const int img_channels = static_cast<int>(in_x_dim[1]);
  const int img_height = static_cast<int>(in_x_dim[2]);
  const int img_width = static_cast<int>(in_x_dim[3]);

  int output_height = Im2SeqOutputSize(img_height, kernels[0], paddings[0],
                                       paddings[2], strides[0]);
  int output_width = Im2SeqOutputSize(img_width, kernels[1], paddings[1],
                                      paddings[3], strides[1]);

  out->mutable_data<float>({batch_size * output_height * output_width,
                            img_channels * kernels[0] * kernels[1]});
  const std::vector<int> dilations({1, 1});
  // TODO(): verify
  auto out_dims = out->dims();
  out->Resize({batch_size, out->numel() / batch_size});
  for (int i = 0; i < batch_size; i++) {
    const Tensor src =
        in_x->Slice(i, i + 1).Resize({img_channels, img_height, img_width});
    Tensor dst = out->Slice(i, i + 1).Resize(
        {output_height, output_width, img_channels, kernels[0], kernels[1]});
    math::Im2ColFunctor<math::ColFormat::kOCF, CPU, float> f;
    f(src, dilations, strides, paddings, &dst);
  }
  out->Resize(out_dims);
  framework::LoD lod(1);
  lod[0].reserve(batch_size + 1);
  int offset = 0;
  lod[0].push_back(offset);
  for (int i = 0; i < batch_size; ++i) {
    offset += output_height * output_width;
    lod[0].push_back(offset);
  }
  out->set_lod(lod);
}

template class Im2SequenceKernel<CPU, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
