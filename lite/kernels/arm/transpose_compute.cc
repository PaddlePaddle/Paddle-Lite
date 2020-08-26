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

#include "lite/kernels/arm/transpose_compute.h"
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename Dtype>
void transpose_mat(const Dtype* din,
                   Dtype* dout,
                   const int num,
                   const int width,
                   const int height);
void transpose_mat(const float* din,
                   float* dout,
                   const int num,
                   const int width,
                   const int height) {
  int nw = width >> 2;
  int nh = height >> 2;
  int size_in = width * height;

  for (int i = 0; i < num; ++i) {
    float* ptr_out = dout + i * size_in;
    const float* ptr_in = din + i * size_in;
#pragma omp parallel for
    for (int h = 0; h < nh; h++) {
      const float* ptr_din_row = ptr_in + h * 4 * width;
      for (int w = 0; w < nw; w++) {
        float* data_out_ptr = ptr_out + w * 4 * height + h * 4;
        const float* din0 = ptr_din_row;
        const float* din1 = din0 + width;
        const float* din2 = din1 + width;
        const float* din3 = din2 + width;

        float* dout0 = data_out_ptr;
        float* dout1 = dout0 + height;
        float* dout2 = dout1 + height;
        float* dout3 = dout2 + height;
#ifdef __aarch64__
        float32x4_t vr0 = vld1q_f32(din0);
        float32x4_t vr1 = vld1q_f32(din1);
        float32x4_t vr2 = vld1q_f32(din2);
        float32x4_t vr3 = vld1q_f32(din3);
        float32x4_t re0 = vtrn1q_f32(vr0, vr1);
        float32x4_t re1 = vtrn2q_f32(vr0, vr1);
        float32x4_t re2 = vtrn1q_f32(vr2, vr3);
        float32x4_t re3 = vtrn2q_f32(vr2, vr3);
        vst1_f32(dout0, vget_low_f32(re0));
        dout0 += 2;
        vst1_f32(dout0, vget_low_f32(re2));
        vst1_f32(dout1, vget_low_f32(re1));
        dout1 += 2;
        vst1_f32(dout1, vget_low_f32(re3));
        vst1_f32(dout2, vget_high_f32(re0));
        dout2 += 2;
        vst1_f32(dout2, vget_high_f32(re2));
        vst1_f32(dout3, vget_high_f32(re1));
        dout3 += 2;
        vst1_f32(dout3, vget_high_f32(re3));
#else
        asm("vld1.32 {d0, d1}, [%[in0]]    \n"
            "vld1.32 {d2, d3}, [%[in1]]    \n"
            "vld1.32 {d4, d5}, [%[in2]]    \n"
            "vld1.32 {d6, d7}, [%[in3]]    \n"
            "vtrn.32 q0, q1                \n"
            "vtrn.32 q2, q3                \n"
            "vswp d1, d4                   \n"
            "vswp d3, d6                   \n"
            "vst1.32 {d0, d1}, [%[out0]]   \n"
            "vst1.32 {d2, d3}, [%[out1]]   \n"
            "vst1.32 {d4, d5}, [%[out2]]   \n"
            "vst1.32 {d6, d7}, [%[out3]]   \n"
            :
            : [out0] "r"(dout0),
              [out1] "r"(dout1),
              [out2] "r"(dout2),
              [out3] "r"(dout3),
              [in0] "r"(din0),
              [in1] "r"(din1),
              [in2] "r"(din2),
              [in3] "r"(din3)
            : "q0", "q1", "q2", "q3");
#endif
        ptr_din_row += 4;
      }
    }
    // remian
    for (int h = 0; h < height; h++) {
      for (int w = nw * 4; w < width; w++) {
        const float* data_in_ptr = ptr_in + h * width + w;
        float* data_out_ptr = ptr_out + w * height + h;
        *data_out_ptr = *data_in_ptr;
      }
    }
    for (int w = 0; w < width; w++) {
      for (int h = nh * 4; h < height; h++) {
        const float* data_in_ptr = ptr_in + h * width + w;
        float* data_out_ptr = ptr_out + w * height + h;
        *data_out_ptr = *data_in_ptr;
      }
    }
  }
}

std::vector<int> get_stride(const paddle::lite::DDimLite& dims) {
  std::vector<int> data_stride{0};

  for (int i = 0; i < dims.size(); ++i) {
    data_stride.push_back(dims.count(i, dims.size()));
  }
  return data_stride;
}

void TransposeCompute::PrepareForRun() {
  auto& param = Param<operators::TransposeParam>();
  auto* input = param.x;
  auto* output = param.output;

  int _num_axes = input->dims().size();
  CHECK(_num_axes == param.axis.size())
      << "axis size is not match to input dims";

  need_trans = false;
  for (int i = 0; i < _num_axes; ++i) {
    if (param.axis[i] != i) {
      need_trans = true;
      break;
    }
  }

  if (!need_trans) {
    return;
  }

  std::vector<int> axis_diff;
  int j = 0;
  for (int i = 0; i < _num_axes; ++i) {
    if (param.axis[j] != i) {
      axis_diff.push_back(j);
    } else {
      j++;
    }
  }
  for (int i = 0; i < axis_diff.size(); i++) {
  }
  if (input->dims().count(axis_diff[0], _num_axes) == 1) {
    need_trans = false;
    return;
  }

  if (axis_diff.size() == 1) {
    trans_mat = true;
    _trans_num = input->dims().count(0, std::max(axis_diff[0], 0));
    _trans_w = input->dims().count(axis_diff[0] + 1, _num_axes);
    _trans_h = input->dims()[axis_diff[0]];

  } else {
    trans_mat = false;
    _new_steps = get_stride(output->dims());
    _old_steps = get_stride(input->dims());
  }
}

template <typename Dtype>
void TransposeCompute_(const std::vector<int>& axis,
                       const lite::Tensor* input,
                       lite::Tensor* output) {
  // const Dtype *input_ptr = input->data<Dtype>();
  const Dtype* input_ptr = input->data<float>();
  Dtype* output_ptr = output->mutable_data<Dtype>();

  // input and output's shape dimension must >= 2 && <= 6.
  const DDim& in_dim = input->dims();
  const DDim& out_dim = output->dims();

  // precompute inverted output dim and strides
  size_t rout_dim[6], strides[6];
  int permute = axis.size();  // permute must >=2 && <= 6.
  for (int i = 0; i < permute; ++i) {
    int k = permute - 1 - i;
    strides[k] = 1;
    for (int j = axis[i] + 1; j < permute; ++j) {
      strides[k] *= in_dim[j];
    }
    rout_dim[k] = out_dim[i];
  }

  // unroll the first 2 dimensions
  int reamin_dim = 1;
  for (int i = 2; i < out_dim.size(); ++i) {
    reamin_dim *= out_dim[i];
  }

#pragma omp parallel for collapse(2)
  for (int batch = 0; batch < out_dim[0]; ++batch) {
    for (int j = 0; j < out_dim[1]; ++j) {
      size_t offset = batch * strides[permute - 1] + j * strides[permute - 2];
      Dtype* out_ptr = output_ptr + (batch * out_dim[1] + j) * reamin_dim;
      int indics[4] = {0, 0, 0, 0};
      for (int k = 0; k < reamin_dim; ++k) {
        out_ptr[k] = input_ptr[offset];
        indics[0] += 1;
        offset += strides[0];
        for (int p = 0; p < permute - 3; ++p) {
          if (indics[p] == rout_dim[p]) {
            indics[p + 1] += 1;
            indics[p] = 0;
            offset += strides[p + 1];
            offset -= rout_dim[p] * strides[p];
          } else {
            break;
          }
        }
      }
    }
  }
}
// Transpose
void TransposeCompute::Run() {
  auto& param = Param<operators::TransposeParam>();
  auto* input = param.x;
  auto* output = param.output;
  const std::vector<int> axis = param.axis;

  //! only copy the data
  if (!need_trans) {
    output->CopyDataFrom(*input);
    return;
  }

  const float* din = static_cast<const float*>(input->data<float>());
  float* dout = static_cast<float*>(output->mutable_data<float>());
  //! transpose the data
  if (trans_mat) {
    transpose_mat(din, dout, _trans_num, _trans_w, _trans_h);
  } else {
    TransposeCompute_<float>(axis, input, output);
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

// Transpose
REGISTER_LITE_KERNEL(transpose,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::TransposeCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();

// Transpose2
REGISTER_LITE_KERNEL(transpose2,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::Transpose2Compute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("XShape", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
