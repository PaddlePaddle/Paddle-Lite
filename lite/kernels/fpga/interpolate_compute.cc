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

#include "lite/kernels/fpga/interpolate_compute.h"
#include <string>
#include <vector>
#include "lite/backends/fpga/KD/debugger.hpp"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace fpga {

using float16 = zynqmp::float16;

void BilinearInterpCompute::Run() {
  auto& param = Param<operators::InterpolateParam>();
  zynqmp::Tensor* input_x = param.X->ZynqTensor();

  int out_w = param.Out->dims()[3];
  int out_h = param.Out->dims()[2];
  auto batch_size = param.X->dims()[0];
  auto channels = param.X->dims()[1];
  auto in_h = param.X->dims()[2];
  auto in_w = param.X->dims()[3];

  if (param.OutSize != nullptr) {
    int* new_data = param.OutSize->ZynqTensor()->data<int32_t>();
    out_h = new_data[0];
    out_w = new_data[1];
  }

  zynqmp::Tensor input_float;
  input_float.setAligned(input_x->aligned());
  input_float.setDataLocation(zynqmp::CPU);
  float* input = input_float.mutableData<float>(zynqmp::FP32, input_x->shape());
  input_float.copyFrom(input_x);
  input_float.invalidate();
  input_float.unalignImage();

  zynqmp::Tensor out_float;
  zynqmp::Shape shape(zynqmp::NHWC, {batch_size, out_h, out_w, channels});
  float* output = out_float.mutableData<float>(zynqmp::FP32, shape);

  auto in_hw = in_h * in_w;
  auto out_hw = out_h * out_w;
  auto in_chw = channels * in_hw;
  auto out_chw = channels * out_hw;

  float ratio_h =
      (out_h > 1) ? static_cast<float>(in_h - 1) / (out_h - 1) : 0.f;
  float ratio_w =
      (out_w > 1) ? static_cast<float>(in_w - 1) / (out_w - 1) : 0.f;

  if (in_h == out_h && in_w == out_w) {
    memcpy(output, input, param.X->numel() * sizeof(float));
  } else {
    // #pragma omp parallel for
    for (int k = 0; k < batch_size; ++k) {  // loop for batches
      for (int i = 0; i < out_h; ++i) {     // loop for images
        int h = ratio_h * i;
        int hid = (h < in_h - 1) ? in_w * channels : 0;
        float h1lambda = ratio_h * i - h;
        float h2lambda = 1.f - h1lambda;

        for (int j = 0; j < out_w; ++j) {
          int w = ratio_w * j;
          int wid = (w < in_w - 1) ? channels : 0;
          float w1lambda = ratio_w * j - w;
          float w2lambda = 1.f - w1lambda;
          // calculate four position for bilinear interpolation
          const float* in_pos =
              &input[(k * in_h * in_w + h * in_w + w) * channels];
          float* out_pos =
              &output[(k * out_w * out_h + i * out_w + j) * channels];
          for (int c = 0; c < channels; ++c) {  // loop for channels
            // bilinear interpolation
            out_pos[c] = static_cast<float>(
                h2lambda *
                    (w2lambda * in_pos[0 + c] + w1lambda * in_pos[wid + c]) +
                h1lambda * (w2lambda * in_pos[hid + c] +
                            w1lambda * in_pos[hid + wid + c]));
          }
        }
      }
    }
  }
  out_float.flush();
  param.Out->mutable_data<float16>();
  param.Out->ZynqTensor()->setDataLocation(zynqmp::CPU);
  param.Out->ZynqTensor()->copyFrom(&out_float);
  param.Out->ZynqTensor()->flush();

  Debugger::get_instance().registerOutput("bilinear_interp",
                                          param.Out->ZynqTensor());
}

void nearest_interp(const float16* src,
                    int w_in,
                    int h_in,
                    int c,
                    float16* dst,
                    int w_out,
                    int h_out,
                    float scale_x,
                    float scale_y,
                    bool with_align) {
  float scale_w_new = (with_align)
                          ? (static_cast<float>(w_in - 1) / (w_out - 1))
                          : (static_cast<float>(w_in) / (w_out));
  float scale_h_new = (with_align)
                          ? (static_cast<float>(h_in - 1) / (h_out - 1))
                          : (static_cast<float>(h_in) / (h_out));
  if (with_align) {
    for (int h = 0; h < h_out; ++h) {
      float16* dst_p = dst + h * w_out * c;
      int near_y = static_cast<int>(scale_h_new * h + 0.5);
      for (int w = 0; w < w_out; ++w) {
        int near_x = static_cast<int>(scale_w_new * w + 0.5);
        const float16* src_n = src + (near_y * w_in + near_x) * c;
        memcpy(dst_p, src_n, c * sizeof(float16));
        dst_p += c;
      }
    }
  } else {
    for (int h = 0; h < h_out; ++h) {
      float16* dst_p = dst + h * w_out;
      int near_y = static_cast<int>(scale_h_new * h);
      for (int w = 0; w < w_out; ++w) {
        int near_x = static_cast<int>(scale_w_new * w);

        const float16* src_n = src + (near_y * w_in + near_x) * c;
        memcpy(dst_p, src_n, c * sizeof(float16));
        dst_p += c;
      }
    }
  }
}

void NearestInterpCompute::PrepareForRun() {
  auto& param = Param<operators::InterpolateParam>();
  lite::Tensor* X = param.X;
  lite::Tensor* OutSize = param.OutSize;
  lite::Tensor* Out = param.Out;
  Out->mutable_data<float16>();

  zynqmp::ResizeParam& norm_param = pe_.param();
  norm_param.input = X->ZynqTensor();
  norm_param.output = Out->ZynqTensor();

  pe_.init();
  pe_.apply();
}

inline std::vector<int> get_new_shape(
    std::vector<const lite::Tensor*> list_new_shape_tensor) {
  // get tensor from
  std::vector<int> vec_new_shape;
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    vec_new_shape.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
  }
  return vec_new_shape;
}

template <typename T>
inline std::vector<T> get_new_data_from_tensor(const Tensor* new_data_tensor) {
  std::vector<T> vec_new_data;
  auto* new_data = new_data_tensor->data<T>();
  lite::Tensor cpu_starts_tensor;
  vec_new_data =
      std::vector<T>(new_data, new_data + new_data_tensor->dims().production());
  return vec_new_data;
}

void interpolate(lite::Tensor* X,
                 lite::Tensor* OutSize,
                 std::vector<const lite::Tensor*> SizeTensor,
                 lite::Tensor* Scale,
                 lite::Tensor* Out,
                 int out_height,
                 int out_width,
                 float scale,
                 bool with_align,
                 std::string interpolate_type) {
  int in_h = X->dims()[2];
  int in_w = X->dims()[3];
  if (SizeTensor.size() > 0) {
    auto new_size = get_new_shape(SizeTensor);
    out_height = new_size[0];
    out_width = new_size[1];
  } else {
    auto scale_tensor = Scale;
    if (scale_tensor != nullptr) {
      auto scale_data = get_new_data_from_tensor<float>(scale_tensor);
      scale = scale_data[0];
    }
    if (scale > 0) {
      out_height = static_cast<int>(in_h * scale);
      out_width = static_cast<int>(in_w * scale);
    }
    auto out_size = OutSize;
    if (out_size != nullptr) {
      auto out_size_data = get_new_data_from_tensor<int>(out_size);
      out_height = out_size_data[0];
      out_width = out_size_data[1];
    }
  }
  float height_scale = scale;
  float width_scale = scale;
  if (out_width > 0 && out_height > 0) {
    height_scale = static_cast<float>(out_height / X->dims()[2]);
    width_scale = static_cast<float>(out_width / X->dims()[3]);
  }
  int num_cout = X->dims()[0];
  int c_cout = X->dims()[1];
  Out->Resize({num_cout, c_cout, out_height, out_width});

  float16* dout = Out->mutable_data<float16>();
  const float16* din = X->data<float16>();
  int out_num = Out->dims()[0];
  int out_c = Out->dims()[1];
  int count = out_num;
  int out_h = Out->dims()[2];
  int out_w = Out->dims()[3];
  int spatial_in = in_h * in_w;
  int spatial_out = out_h * out_w;

  for (int i = 0; i < count; ++i) {
    nearest_interp(din + spatial_in * i,
                   in_w,
                   in_h,
                   out_c,
                   dout + spatial_out * i,
                   out_w,
                   out_h,
                   1.f / width_scale,
                   1.f / height_scale,
                   with_align);
  }
}

void NearestInterpCompute::Run() {
  auto& param = Param<operators::InterpolateParam>();
  lite::Tensor* X = param.X;
  lite::Tensor* OutSize = param.OutSize;
  auto SizeTensor = param.SizeTensor;
  auto Scale = param.Scale;
  lite::Tensor* Out = param.Out;
  float scale = param.scale;
  int out_w = param.out_w;
  int out_h = param.out_h;
  bool align_corners = param.align_corners;

  std::string interp_method = "";

  X->ZynqTensor()->syncToCPU();
  interpolate(X,
              OutSize,
              SizeTensor,
              Scale,
              Out,
              out_h,
              out_w,
              scale,
              align_corners,
              interp_method);

  Out->ZynqTensor()->flush();
  Out->ZynqTensor()->copyScaleFrom(X->ZynqTensor());
}

} /* namespace fpga */
} /* namespace kernels */
} /* namespace lite */
} /* namespace paddle */

REGISTER_LITE_KERNEL(bilinear_interp,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::BilinearInterpCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kFPGA), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();

REGISTER_LITE_KERNEL(nearest_interp,
                     kFPGA,
                     kFP16,
                     kNHWC,
                     paddle::lite::kernels::fpga::NearestInterpCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kFPGA),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kNHWC))})
    .BindInput("OutSize",
               {LiteType::GetTensorTy(TARGET(kFPGA), PRECISION(kInt32))})
    .BindInput("SizeTensor",
               {LiteType::GetTensorTy(TARGET(kARM), PRECISION(kInt32))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kFPGA),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kNHWC))})
    .Finalize();
