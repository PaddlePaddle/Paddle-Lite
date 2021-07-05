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
#pragma once

#include <Eigen/Core>
#include <string>
#include <vector>
#include "lite/backends/x86/math/blas.h"
#include "lite/backends/x86/math/conv_bias.h"
#ifdef LITE_WITH_AVX
#include "lite/backends/x86/math/conv_utils.h"
#endif
#include "lite/backends/x86/math/im2col.h"
#include "lite/backends/x86/math/vol2col.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/core/types.h"
#include "lite/operators/conv_op.h"

namespace paddle {
namespace lite_metal {
namespace kernels {
namespace x86 {

inline bool IsExpand(const std::vector<int64_t>& filter_dim,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings,
                     const std::vector<int>& dilations) {
  bool filter_1 = true, strides_1 = true, padding_0 = true, dilation_1 = true;
  for (size_t j = 0; j < strides.size(); ++j) {
    filter_1 = filter_1 && (static_cast<int>(filter_dim[j + 2]) == 1);
    strides_1 = strides_1 && (strides[j] == 1);
    padding_0 = padding_0 && (paddings[j] == 0);
    dilation_1 = dilation_1 && (dilations[j] == 1);
  }
  return !(filter_1 && strides_1 && padding_0 && dilation_1);
}

template <typename T>
class Conv2dCompute : public KernelLite<TARGET(kX86), PRECISION(kFloat)> {
 public:
  virtual void PrepareForRun();

  virtual void ReInitWhenNeeded() {
    if (impl_) {
      impl_->ReInitWhenNeeded();
    }
  }

  virtual void Run() {
    if (impl_) {
      return impl_->Run();
    }
    // To-do(qili93): remove below lines of code after all kernels implemented
    auto& context = ctx_->As<X86Context>();
    auto& param = *param_.get_mutable<operators::ConvParam>();
    lite_metal::Tensor filter = *param.filter;
    param.output->template mutable_data<T>();
    const int batch_size = static_cast<int>(param.x->dims()[0]);

    const int kh = static_cast<int>(param.filter->dims()[2]);
    const int kw = static_cast<int>(param.filter->dims()[3]);

    const int sh = static_cast<int>(param.strides[0]);
    const int sw = static_cast<int>(param.strides[1]);

    auto paddings = *param.paddings;
    const int ph = paddings[0];
    const int pw = paddings[2];

    bool kps_equal = (pw == ph) && (sw == sh) && (kw == kh);
    bool pads_equal =
        ((paddings[0] == paddings[1]) && (paddings[2] == paddings[3]));
    bool flag_1x1gemm = false;
    if (kw == 1 && sw == 1 && pw == 0 && kps_equal && pads_equal) {
      flag_1x1gemm = true;
    }

    std::vector<int64_t> filter_shape_vec(filter.dims().Vectorize());
    std::vector<int64_t> output_shape_vec(param.output->dims().Vectorize());
    size_t data_dim = filter_shape_vec.size() - 2;
    std::vector<int64_t> col_shape_vec(1 + 2 * data_dim);
    col_shape_vec[0] = param.x->dims()[1] / param.groups;
    for (size_t j = 0; j < data_dim; ++j) {
      col_shape_vec[j + 1] = filter_shape_vec[j + 2];
      col_shape_vec[j + 1 + data_dim] = output_shape_vec[j + 2];
    }
    lite_metal::DDim col_shape(col_shape_vec);
    lite_metal::DDim col_matrix_shape = col_shape.Flatten2D(data_dim + 1);
    bool is_expand = IsExpand(
        filter_shape_vec, param.strides, *param.paddings, *param.dilations);
    lite_metal::Tensor col;
    lite_metal::Tensor col_matrix;
    if (is_expand) {
      col.Resize(col_shape);
      col.mutable_data<T>();
      col_matrix.ShareDataWith(col);
      col_matrix.Resize(col_matrix_shape);
    }
    lite_metal::DDim input_shape = param.x->dims().Slice(1, param.x->dims().size());
    lite_metal::DDim filter_matrix_shape(std::vector<int64_t>{
        filter.dims()[0], filter.dims().production() / filter.dims()[0]});
    filter.Resize(filter_matrix_shape);
    lite_metal::DDim output_matrix_shape(std::vector<int64_t>{
        param.output->dims()[1],
        param.output->dims().production() /
            (param.output->dims()[0] * param.output->dims()[1])});
    int in_step = static_cast<int>(param.x->dims()[1]) / param.groups;
    int out_step = static_cast<int>(param.output->dims()[1]) / param.groups;
    paddle::lite_metal::x86::math::Vol2ColFunctor<lite_metal::TargetType::kX86, T> vol2col;
    paddle::lite_metal::x86::math::Im2ColFunctor<
        paddle::lite_metal::x86::math::ColFormat::kCFO,
        lite_metal::TargetType::kX86,
        T>
        im2col;
    auto blas =
        paddle::lite_metal::x86::math::GetBlas<lite_metal::TargetType::kX86, T>(context);
    for (int i = 0; i < batch_size; i++) {
      lite_metal::Tensor in_batch = param.x->template Slice<T>(i, i + 1);
      in_batch.Resize(input_shape);
      lite_metal::Tensor out_batch = param.output->template Slice<T>(i, i + 1);
      out_batch.Resize(output_matrix_shape);
      for (int g = 0; g < param.groups; g++) {
        lite_metal::Tensor in_slice =
            in_batch.Slice<T>(static_cast<int64_t>(g * in_step),
                              static_cast<int64_t>((g + 1) * in_step));
        auto paddings = *param.paddings;
        if (!is_expand) {
          col.ShareDataWith(in_slice);
          col_matrix.ShareDataWith(col);
          col_matrix.Resize(col_matrix_shape);
        } else if (data_dim == 2U && !flag_1x1gemm) {
          // im2col
          im2col(context,
                 in_slice,
                 *param.dilations,
                 param.strides,
                 std::vector<int>{
                     paddings[0], paddings[2], paddings[0], paddings[2]},
                 &(col));
        } else if (data_dim == 3U) {
          // vol2col
          vol2col(context,
                  in_slice,
                  *param.dilations,
                  param.strides,
                  *param.paddings,
                  &(col));
        }

        // gemm
        lite_metal::Tensor out_slice;
        out_slice =
            out_batch.Slice<T>(static_cast<int64_t>(g * out_step),
                               static_cast<int64_t>((g + 1) * out_step));
        lite_metal::Tensor filter_slice;
        filter_slice =
            filter.Slice<T>(static_cast<int64_t>(g * out_step),
                            static_cast<int64_t>((g + 1) * out_step));
        blas.MatMul(filter_slice,
                    false,
                    col_matrix,
                    false,
                    T(1.0),
                    &(out_slice),
                    T(0.0));
      }
    }

    // for bias
    if (param.bias) {
      const int output_channel = static_cast<int>(param.output->dims()[1]);
      const int output_number =
          param.output->dims().production() /
          (param.output->dims()[0] * param.output->dims()[1]);
      auto* bias_data = param.bias->template data<T>();
      auto* out_data = param.output->template mutable_data<T>();
      auto act_param = param.activation_param;
      if (act_param.has_active) {
        if (act_param.active_type == lite_metal_api::ActivationType::kRelu) {
          lite_metal::x86::math::bias_add_relu_broadcast(out_data,
                                                   bias_data,
                                                   out_data,
                                                   batch_size,
                                                   output_channel,
                                                   output_number);
        } else if (act_param.active_type == lite_metal_api::ActivationType::kRelu6) {
          lite_metal::x86::math::bias_add_relu6_broadcast(out_data,
                                                    bias_data,
                                                    out_data,
                                                    batch_size,
                                                    output_channel,
                                                    output_number);
        } else {
          LOG(FATAL) << "[X86] unsupported Activation type";
        }
      } else {
        lite_metal::x86::math::bias_add_broadcast(out_data,
                                            bias_data,
                                            out_data,
                                            batch_size,
                                            output_channel,
                                            output_number);
      }
    }
  }

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite_metal::profile::OpCharacter* ch) {
    ch->kernel_func_name = "NotImplForConv";
  }
#endif

  ~Conv2dCompute() {
    if (impl_ != nullptr) {
      delete impl_;
    }
  }

 private:
  using param_t = operators::ConvParam;
  KernelLite<TARGET(kX86), PRECISION(kFloat)>* impl_{nullptr};
};

}  // namespace x86
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
