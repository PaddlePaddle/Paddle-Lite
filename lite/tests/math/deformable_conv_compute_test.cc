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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include "lite/core/context.h"
#include "lite/core/profile/timer.h"
#include "lite/operators/op_params.h"
#include "lite/tests/utils/naive_math_impl.h"
#include "lite/tests/utils/tensor_utils.h"

#ifdef LITE_WITH_ARM
#include "lite/kernels/arm/deformable_conv_compute.h"
#endif  // LITE_WITH_ARM

DEFINE_int32(power_mode,
             3,
             "power mode: "
             "0 for POWER_HIGH;"
             "1 for POWER_LOW;"
             "2 for POWER_FULL;"
             "3 for NO_BIND");
DEFINE_int32(threads, 1, "threads num");
DEFINE_int32(warmup, 0, "warmup times");
DEFINE_int32(repeats, 1, "repeats times");
DEFINE_bool(basic_test, false, "do all tests");
DEFINE_bool(check_result, true, "check the result");

DEFINE_int32(batch, 1, "batch size");
DEFINE_int32(in_channel, 32, "input channel");
DEFINE_int32(in_height, 112, "input height");
DEFINE_int32(in_width, 112, "input width");

DEFINE_int32(out_channel, 32, "output channel");
DEFINE_int32(group, 1, "group");
DEFINE_int32(kernel_h, 3, "kernel height");
DEFINE_int32(kernel_w, 3, "kernel width");
DEFINE_int32(pad_h, 1, "pad height");
DEFINE_int32(pad_w, 1, "pad width");
DEFINE_int32(stride_h, 1, "stride height");
DEFINE_int32(stride_w, 1, "stride width");
DEFINE_int32(dila_h, 1, "dilation height");
DEFINE_int32(dila_w, 1, "dilation width");

DEFINE_int32(flag_act,
             0,
             "do activation");  // 0-no act, 1-relu, 2-relu6, 4-leakyrelu
DEFINE_double(leakey_relu_alpha, 1.0, "leakey relu alpha");
DEFINE_bool(flag_bias, true, "with bias");

typedef paddle::lite_metal::DDim DDim;
typedef paddle::lite_metal::Tensor Tensor;
typedef paddle::lite_metal::operators::DeformableConvParam DeformableConvParam;
typedef paddle::lite_metal::operators::ActivationParam ActivationParam;

using paddle::lite_metal::profile::Timer;

DDim compute_out_dim(const DDim& dim_in,
                     const paddle::lite_metal::operators::ConvParam& param) {
  DDim dim_out = dim_in;
  auto paddings = *param.paddings;
  auto dilations = *param.dilations;
  dim_out[1] = param.filter->dims()[0];
  auto kernel_h = param.filter->dims()[2];
  auto kernel_w = param.filter->dims()[3];
  auto h = dim_in[2];
  auto w = dim_in[3];
  int dila_h = dilations[0];
  int dila_w = dilations[1];
  int pad_top = paddings[0];
  int pad_bottom = paddings[1];
  int pad_left = paddings[2];
  int pad_right = paddings[3];
  int stride_h = param.strides[0];
  int stride_w = param.strides[1];
  auto kernel_exten = dila_h * (kernel_h - 1) + 1;
  auto hout = (h + pad_top + pad_bottom - kernel_exten) / stride_h + 1;
  kernel_exten = dila_w * (kernel_w - 1) + 1;
  auto wout = (w + pad_left + pad_right - kernel_exten) / stride_w + 1;
  dim_out[2] = hout;
  dim_out[3] = wout;
  return dim_out;
}

template <class T>
static void MatMul(const Tensor& mat_a,
                   const Tensor& mat_b,
                   T alpha,
                   Tensor* mat_out,
                   T beta,
                   const T* bias,
                   bool flag_bias,
                   bool flag_relu) {
  auto dim_a = mat_a.dims();
  auto dim_b = mat_b.dims();
  auto dim_out = mat_out->dims();

  int M = dim_out[0];
  int N = dim_out[1];
  int K = dim_a[1];
  auto* pA = mat_a.data<T>();
  auto* pB = mat_b.data<T>();
  auto* pC = mat_out->mutable_data<T>();
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      T sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += pA[i * K + k] * pB[k * N + j];
      }
      pC[i * N + j] = sum * alpha + beta;
      if (flag_bias) {
        pC[i * N + j] += bias[i];
      }
      if (flag_relu) {
        pC[i * N + j] = pC[i * N + j] > 0 ? pC[i * N + j] : 0;
      }
    }
  }
}

template <typename T>
static T DmcnIm2colBilinear(const T* bottom_data,
                            const int data_width,
                            const int height,
                            const int width,
                            T h,
                            T w) {
  int h_low = floor(h);
  int w_low = floor(w);
  int h_high = h_low + 1;
  int w_high = w_low + 1;

  T lh = h - h_low;
  T lw = w - w_low;
  T hh = 1 - lh;
  T hw = 1 - lw;

  T v1 =
      (h_low >= 0 && w_low >= 0) ? bottom_data[h_low * data_width + w_low] : 0;
  T v2 = (h_low >= 0 && w_high <= width - 1)
             ? bottom_data[h_low * data_width + w_high]
             : 0;
  T v3 = (h_high <= height - 1 && w_low >= 0)
             ? bottom_data[h_high * data_width + w_low]
             : 0;
  T v4 = (h_high <= height - 1 && w_high <= width - 1)
             ? bottom_data[h_high * data_width + w_high]
             : 0;

  T w1 = hh * hw;
  T w2 = hh * lw;
  T w3 = lh * hw;
  T w4 = lh * lw;

  return w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4;
}

template <typename T>
static void ModulatedDeformableIm2colCPUKernel(
    const int num_kernels,
    const T* data_im,
    const T* data_offset,
    const T* data_mask,
    const int height,
    const int width,
    const int kernel_h,
    const int kernel_w,
    const int pad_h,
    const int pad_w,
    const int stride_h,
    const int stride_w,
    const int dilation_h,
    const int dilation_w,
    const int channel_per_deformable_group,
    const int batch_size,
    const int num_channels,
    const int deformable_group,
    const int height_col,
    const int width_col,
    T* data_col) {
  for (int i = 0; i < num_kernels; i++) {
    const int w_col = i % width_col;
    const int h_col = (i / width_col) % height_col;
    const int b_col = (i / width_col) / height_col % batch_size;
    const int c_im = (i / width_col / height_col) / batch_size;
    const int c_col = c_im * kernel_h * kernel_w;

    const int deformable_group_index = c_im / channel_per_deformable_group;

    const int h_in = h_col * stride_h - pad_h;
    const int w_in = w_col * stride_w - pad_w;

    T* data_col_ptr =
        data_col +
        ((c_col * batch_size + b_col) * height_col + h_col) * width_col + w_col;
    const T* data_im_ptr =
        data_im + (b_col * num_channels + c_im) * height * width;
    const T* data_offset_ptr =
        data_offset +
        (b_col * deformable_group + deformable_group_index) * 2 * kernel_h *
            kernel_w * height_col * width_col;
    const T* data_mask_ptr =
        data_mask +
        (b_col * deformable_group + deformable_group_index) * kernel_h *
            kernel_w * height_col * width_col;

    for (int i = 0; i < kernel_h; ++i) {
      for (int j = 0; j < kernel_w; ++j) {
        const int data_offset_h_ptr =
            ((2 * (i * kernel_w + j)) * height_col + h_col) * width_col + w_col;
        const int data_offset_w_ptr =
            ((2 * (i * kernel_w + j) + 1) * height_col + h_col) * width_col +
            w_col;
        const int data_mask_hw_ptr =
            ((i * kernel_w + j) * height_col + h_col) * width_col + w_col;

        const T offset_h = data_offset_ptr[data_offset_h_ptr];
        const T offset_w = data_offset_ptr[data_offset_w_ptr];
        const T mask = data_mask_ptr[data_mask_hw_ptr];
        T val = static_cast<T>(0);
        const T h_im = h_in + i * dilation_h + offset_h;
        const T w_im = w_in + j * dilation_w + offset_w;
        if (h_im > -1 && w_im > -1 && h_im < height && w_im < width) {
          val =
              DmcnIm2colBilinear(data_im_ptr, width, height, width, h_im, w_im);
        }
        *data_col_ptr = val * mask;
        data_col_ptr += batch_size * height_col * width_col;
      }
    }
  }
}

template <typename T>
static inline void ModulatedDeformableIm2colCPU(
    const T* data_im,
    const T* data_offset,
    const T* data_mask,
    const std::vector<int64_t> im_shape,
    const std::vector<int64_t> col_shape,
    const std::vector<int64_t> filter_shape,
    const std::vector<int> paddings,
    const std::vector<int> strides,
    const std::vector<int> dilations,
    const int deformable_groups,
    T* data_col) {
  int channel_per_deformable_group = im_shape[0] / deformable_groups;
  int num_kernels = im_shape[0] * col_shape[1] * col_shape[2] * col_shape[3];

  // get outputs of im2col with offset by bilinear interpolation
  ModulatedDeformableIm2colCPUKernel(num_kernels,
                                     data_im,
                                     data_offset,
                                     data_mask,
                                     im_shape[1],
                                     im_shape[2],
                                     filter_shape[2],
                                     filter_shape[3],
                                     paddings[0],
                                     paddings[1],
                                     strides[0],
                                     strides[1],
                                     dilations[0],
                                     dilations[1],
                                     channel_per_deformable_group,
                                     col_shape[1],
                                     im_shape[0],
                                     deformable_groups,
                                     col_shape[2],
                                     col_shape[3],
                                     data_col);
}

void deformable_conv_compute_basic(const Tensor* input,
                                   const Tensor* offset,
                                   const Tensor* mask,
                                   Tensor* output,
                                   const Tensor& filter,
                                   const Tensor* bias,
                                   int groups,
                                   int deformable_groups,
                                   int im2col_step,
                                   const std::vector<int>& strides,
                                   const std::vector<int>& paddings,
                                   const std::vector<int>& dilations,
                                   bool flag_relu) {
  const float* bias_data = bias ? bias->data<float>() : nullptr;
  bool is_bias = bias ? true : false;
  const int batch_size = static_cast<int>(input->dims()[0]);
  std::vector<int64_t> filter_shape_vec(filter.dims().Vectorize());
  std::vector<int64_t> output_shape_vec(output->dims().Vectorize());

  // col_shape_vec: {c_i * k_h * k_w, im2col_step, o_h, o_w}
  std::vector<int64_t> col_buffer_shape_vec(filter_shape_vec.size());
  col_buffer_shape_vec[0] =
      input->dims()[1] * filter.dims()[2] * filter.dims()[3];
  col_buffer_shape_vec[1] = im2col_step;
  for (size_t j = 0; j < filter_shape_vec.size() - 2; ++j) {
    col_buffer_shape_vec[j + 2] = output_shape_vec[j + 2];
  }
  DDim col_shape(col_buffer_shape_vec);
  std::vector<int64_t> output_buffer_shape_vec(1);
  output_buffer_shape_vec[0] = batch_size * output_shape_vec[1] *
                               output_shape_vec[2] * output_shape_vec[3];
  DDim output_shape(output_buffer_shape_vec);
  Tensor col_buffer;
  Tensor output_buffer;
  col_buffer.Resize(col_shape);
  col_buffer.mutable_data<float>();
  output_buffer.Resize(output_shape);
  output_buffer.mutable_data<float>();
  int64_t M = output_shape_vec[1] / groups;
  int64_t N = im2col_step * output_shape_vec[2] * output_shape_vec[3];
  int64_t K =
      input->dims()[1] * filter_shape_vec[2] * filter_shape_vec[3] / groups;

  Tensor weight_3d;
  weight_3d.ShareDataWith(filter);
  weight_3d.Resize(DDim({groups, M, K}));
  Tensor col_buffer_3d;
  col_buffer_3d.ShareDataWith(col_buffer);
  col_buffer_3d.Resize(DDim({groups, K, N}));
  Tensor output_4d;
  output_4d.ShareDataWith(output_buffer);
  output_4d.Resize(DDim({batch_size / im2col_step, groups, M, N}));
  output_4d.mutable_data<float>();
  DDim input_shape = input->dims().Slice(1, input->dims().size());
  std::vector<int64_t> input_shape_vec = input_shape.Vectorize();
  int input_dim = input->numel() / input->dims()[0];
  int input_offset_dim = offset->numel() / offset->dims()[0];
  int input_mask_dim = mask->numel() / mask->dims()[0];
  const float* input_ptr = input->data<float>();
  const float* offset_ptr = offset->data<float>();
  const float* mask_ptr = mask->data<float>();
  col_buffer.mutable_data<float>();
  float* col_buffer_ptr = col_buffer.mutable_data<float>();
  for (int i = 0; i < batch_size / im2col_step; ++i) {
    ModulatedDeformableIm2colCPU<float>(
        input_ptr + i * im2col_step * input_dim,
        offset_ptr + i * im2col_step * input_offset_dim,
        mask_ptr + i * im2col_step * input_mask_dim,
        input_shape_vec,
        col_buffer_shape_vec,
        filter_shape_vec,
        paddings,
        strides,
        dilations,
        deformable_groups,
        col_buffer_ptr);
    Tensor output_3d = output_4d.Slice<float>(i, i + 1);
    output_3d.Resize(DDim(output_4d.dims()).Slice(1, output_4d.dims().size()));
    // get the product of pixel and weight
    for (int g = 0; g < groups; ++g) {
      const float* bias_group = bias_data + g * M;
      Tensor weight_3d_slice = weight_3d.Slice<float>(g, g + 1);
      weight_3d_slice.Resize(
          DDim(weight_3d.dims()).Slice(1, weight_3d.dims().size()));
      Tensor col_buffer_3d_slice = col_buffer_3d.Slice<float>(g, g + 1);
      col_buffer_3d_slice.Resize(
          DDim(col_buffer_3d.dims()).Slice(1, col_buffer_3d.dims().size()));
      Tensor output_3d_slice = output_3d.Slice<float>(g, g + 1);
      output_3d_slice.Resize(
          DDim(output_3d.dims()).Slice(1, output_3d.dims().size()));
      MatMul<float>(weight_3d_slice,
                    col_buffer_3d_slice,
                    1.0f,
                    &output_3d_slice,
                    0.0f,
                    bias_group,
                    is_bias,
                    flag_relu);
    }
  }
  output->ShareDataWith(output_buffer);
  output->Resize(DDim(output_shape_vec));
}
#ifdef LITE_WITH_ARM
void test_deformable_conv_fp32(const std::vector<DDim>& input_dims,
                               const DDim& weight_dim,
                               int group,
                               const std::vector<int>& strides,
                               const std::vector<int>& pads,
                               const std::vector<int>& dilas,
                               bool flag_bias,
                               bool flag_relu,
                               bool modulated,
                               const std::vector<int>& thread_num,
                               const std::vector<int>& power_mode,
                               const float leakey_relu_scale) {
#ifdef LITE_WITH_ARM
  paddle::lite_metal::DeviceInfo::Init();
#endif
  DeformableConvParam param;
  param.x = new Tensor;
  param.x->set_precision(PRECISION(kFloat));
  param.conv_param.filter = new Tensor;
  param.conv_param.filter->Resize(weight_dim);
  param.conv_param.filter->set_precision(PRECISION(kFloat));
  param.offset = new Tensor;
  param.offset->set_precision(PRECISION(kFloat));
  param.mask = new Tensor;
  param.mask->set_precision(PRECISION(kFloat));
  if (flag_bias) {
    param.conv_param.bias = new Tensor;
    param.conv_param.bias->Resize({weight_dim[0]});
    param.conv_param.bias->set_precision(PRECISION(kFloat));
  }
  param.conv_param.strides = strides;
  param.conv_param.paddings = std::make_shared<std::vector<int>>(pads);
  param.conv_param.dilations = std::make_shared<std::vector<int>>(dilas);
  param.conv_param.groups = group;
  param.deformable_groups = group;
  param.modulated = modulated;
  const float six = 6.f;
  int flag_act = flag_relu ? 1 : 0;
  if (flag_act > 0) {
    ActivationParam act_param;
    act_param.has_active = true;
    act_param.active_type = (paddle::lite_metal_api::ActivationType)
        flag_act;  // 1-relu, 2-relu6, 4-leakyrelu
    if (flag_act == 1) {
      param.conv_param.fuse_relu = true;
    } else if (flag_act == 2) {
      act_param.Relu_clipped_coef = six;
    } else if (flag_act == 4) {
      act_param.Leaky_relu_alpha = leakey_relu_scale;
    }
    param.conv_param.activation_param = act_param;
  }

  param.output = new Tensor;
  param.output->set_precision(PRECISION(kFloat));

  paddle::lite_metal::fill_tensor_rand(*param.conv_param.filter, -1.f, 1.f);
  //  paddle::lite_metal::fill_tensor_const(*param.filter, 1.f);
  if (flag_bias) {
    paddle::lite_metal::fill_tensor_rand(*param.conv_param.bias, -1.f, 1.f);
    //    paddle::lite_metal::fill_tensor_const(*param.bias, 1.f);
  }
  auto wptr = param.conv_param.filter->data<float>();
  auto bias_ptr = flag_bias ? param.conv_param.bias->data<float>() : nullptr;

  for (auto& cls : power_mode) {
    for (auto& th : thread_num) {
      paddle::lite_metal::kernels::arm::DeformableConvCompute<PRECISION(kFloat),
                                                        PRECISION(kFloat)>
          deformableConv;
      std::unique_ptr<paddle::lite_metal::KernelContext> ctx1(
          new paddle::lite_metal::KernelContext);
      auto& ctx = ctx1->As<paddle::lite_metal::ARMContext>();
      ctx.SetRunMode(static_cast<paddle::lite_metal_api::PowerMode>(cls), th);
      /// set param and context
      for (auto& dim_in : input_dims) {
        param.x->Resize(dim_in);
        DDim out_tmp_dims = compute_out_dim(dim_in, param.conv_param);
        if (out_tmp_dims[2] < 1 || out_tmp_dims[3] < 1) {
          continue;
        }
        param.output->Resize(out_tmp_dims);
        break;
      }
      deformableConv.SetParam(param);
      deformableConv.SetContext(std::move(ctx1));
      /// prepare for run
      deformableConv.PrepareForRun();

      for (auto& dim_in : input_dims) {
        CHECK_EQ(weight_dim[1] * group, dim_in[1])
            << "input channel must equal to weights channel";
        DDim dim_out = compute_out_dim(dim_in, param.conv_param);
        int num = dim_in[0];
        int in_size = dim_in[2] * dim_in[3];
        int kernel_size = weight_dim[2] * weight_dim[3];
        param.offset->Resize(
            {num, 2 * group * kernel_size, dim_in[2], dim_in[3]});
        param.mask->Resize({num, group * kernel_size, dim_in[2], dim_in[3]});
        paddle::lite_metal::fill_tensor_rand(*param.offset, -1.f, 1.f);
        paddle::lite_metal::fill_tensor_rand(*param.mask, -1.f, 1.f);
        if (dim_out[2] < 1 || dim_out[3] < 1) {
          continue;
        }
        if (dim_out[2] != dim_in[2] || dim_out[3] != dim_in[3]) {
          continue;
        }
        param.x->Resize(dim_in);
        param.output->Resize(dim_out);

        paddle::lite_metal::fill_tensor_rand(*param.x, -1.f, 1.f);
        // paddle::lite_metal::fill_tensor_const(*param.x, 1.f);
        auto din = param.x->data<float>();

        Tensor tout_basic;
        if (FLAGS_check_result) {
          auto offset_data = param.offset->data<float>();
          auto mask_data = param.mask->data<float>();
          tout_basic.set_precision(PRECISION(kFloat));
          tout_basic.Resize(dim_out);
          fill_tensor_const(tout_basic, 0.f);
          auto dout_basic = tout_basic.mutable_data<float>();
          LOG(INFO) << "flag_relu: " << flag_relu;
          deformable_conv_compute_basic(param.x,
                                        param.offset,
                                        param.mask,
                                        &tout_basic,
                                        *param.conv_param.filter,
                                        param.conv_param.bias,
                                        param.conv_param.groups,
                                        param.deformable_groups,
                                        param.im2col_step,
                                        param.conv_param.strides,
                                        *param.conv_param.paddings,
                                        *param.conv_param.dilations,
                                        flag_relu);
        }
        /// warm up
        for (int i = 0; i < FLAGS_warmup; ++i) {
          deformableConv.Launch();
        }
        /// compute
        Timer t0;
        for (int i = 0; i < FLAGS_repeats; ++i) {
          t0.Start();
          deformableConv.Launch();
          t0.Stop();
        }

        double gops = 2.0 * dim_out.production() * dim_in[1] * weight_dim[2] *
                      weight_dim[3] / param.conv_param.groups;
        LOG(INFO) << "deformable conv fp32: input shape: " << dim_in
                  << ", output shape" << dim_out
                  << ",running time, avg: " << t0.LapTimes().Avg()
                  << ", min time: " << t0.LapTimes().Min()
                  << ", total GOPS: " << 1e-9 * gops
                  << " GOPS, avg GOPs: " << 1e-6 * gops / t0.LapTimes().Avg()
                  << " GOPs, max GOPs: " << 1e-6 * gops / t0.LapTimes().Min();

        if (FLAGS_check_result) {
          double max_ratio = 0;
          double max_diff = 0;
          tensor_cmp_host(tout_basic, *param.output, max_ratio, max_diff);
          LOG(INFO) << "compare result, max diff: " << max_diff
                    << ", max ratio: " << max_ratio;
          if (std::abs(max_ratio) > 1e-3f) {
            if (max_diff > 5e-4f) {
              LOG(WARNING) << "weights data";
              print_tensor(*param.conv_param.filter);
              LOG(WARNING) << "basic result";
              print_tensor(tout_basic);
              LOG(WARNING) << "lite result";
              print_tensor(*param.output);
              Tensor tdiff;
              tdiff.Resize(tout_basic.dims());
              tdiff.set_precision(PRECISION(kFloat));
              tensor_diff(tout_basic, *param.output, tdiff);
              print_tensor(tdiff);
              LOG(FATAL) << "test fp32 deformable conv: input: " << dim_in
                         << ", output: " << dim_out
                         << ", weight dim: " << weight_dim
                         << ", pad: " << pads[0] << ", " << pads[1] << ", "
                         << pads[2] << ", " << pads[3]
                         << ", stride: " << strides[0] << ", " << strides[1]
                         << ", dila_: " << dilas[0] << ", " << dilas[1]
                         << ", group: " << group
                         << ", bias: " << (flag_bias ? "true" : "false")
                         << ", relu: " << (flag_relu ? "true" : "false")
                         << ", modulated: " << (modulated ? "V2" : "V1")
                         << ", threads: " << th << ", power_mode: " << cls
                         << " failed!!\n";
            }
          }
        }
        LOG(INFO) << "test fp32 deformable conv: input: " << dim_in
                  << ", output: " << dim_out << ", weight dim: " << weight_dim
                  << ", pad: " << pads[0] << ", " << pads[1] << ", " << pads[2]
                  << ", " << pads[3] << ", stride: " << strides[0] << ", "
                  << strides[1] << ", dila_: " << dilas[0] << ", " << dilas[1]
                  << ", group: " << group
                  << ", bias: " << (flag_bias ? "true" : "false")
                  << ", relu: " << (flag_relu ? "true" : "false")
                  << ", modulated: " << (modulated ? "V2" : "V1")
                  << ", threads: " << th << ", power_mode: " << cls
                  << " successed!!\n";
      }
    }
  }

  delete param.x;
  delete param.conv_param.filter;
  delete param.offset;
  delete param.mask;
  delete param.output;
  delete param.conv_param.bias;
}
#else
void test_deformable_conv_fp32(const std::vector<DDim>& input_dims,
                               const DDim& weight_dim,
                               int group,
                               const std::vector<int>& strides,
                               const std::vector<int>& pads,
                               const std::vector<int>& dilas,
                               bool flag_bias,
                               bool flag_relu,
                               bool modulated,
                               const std::vector<int>& thread_num,
                               const std::vector<int>& power_mode,
                               const float leakey_relu_scale) {}
#endif  // LITE_WITH_ARM

#if 1  /// random param conv
TEST(TestDeformableConvRand, test_deformable_conv_rand) {
  if (FLAGS_basic_test) {
    for (auto& cin : {1, 3, 8}) {
      for (auto& cout : {1, 5, 16}) {
        for (auto& g : {1}) {
          for (auto& kw : {1, 2, 3}) {
            for (auto& kh : {1, 2, 3}) {
              for (auto& stride : {1, 2}) {
                for (auto& pad_h : {0, 1, 2}) {
                  for (auto& pad_w : {0, 1, 2}) {
                    for (auto& dila : {1, 2}) {
                      for (auto& modulated : {false, true}) {
                        for (auto& flag_bias : {false, true}) {
                          for (auto& flag_act : {0, 1}) {
                            if (cin % g != 0 || cout % g != 0) {
                              continue;
                            }
                            std::vector<DDim> dims;
                            DDim weights_dim({cout, cin / g, kh, kw});
                            for (auto& batch : {1, 2}) {
                              for (auto& h : {1, 3, 16, 19, 32, 64}) {
                                dims.push_back(DDim({batch, cin, h, h}));
                              }
                            }
                            const float leakey_relu_scale = 8.88;
                            test_deformable_conv_fp32(
                                dims,
                                weights_dim,
                                g,
                                {stride, stride},
                                {pad_h, pad_h, pad_w, pad_w},
                                {dila, dila},
                                flag_bias,
                                flag_act,
                                modulated,
                                {1},
                                {FLAGS_power_mode},
                                leakey_relu_scale);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
#endif  /// random param conv

#if 1  /// custom
TEST(TestDeformableConvCustom, test_deformable_conv_fp32_custom_size) {
  CHECK_EQ(FLAGS_in_channel % FLAGS_group, 0)
      << "input channel must be divided by group";
  CHECK_EQ(FLAGS_out_channel % FLAGS_group, 0)
      << "num_output must be divided by group";
  test_deformable_conv_fp32(
      {DDim({FLAGS_batch, FLAGS_in_channel, FLAGS_in_height, FLAGS_in_width})},
      DDim({FLAGS_out_channel,
            FLAGS_in_channel / FLAGS_group,
            FLAGS_kernel_h,
            FLAGS_kernel_w}),
      FLAGS_group,
      {FLAGS_stride_h, FLAGS_stride_w},
      {FLAGS_pad_h, FLAGS_pad_h, FLAGS_pad_w, FLAGS_pad_w},
      {FLAGS_dila_h, FLAGS_dila_w},
      FLAGS_flag_bias,
      FLAGS_flag_act,
      true,
      {FLAGS_threads},
      {FLAGS_power_mode},
      FLAGS_leakey_relu_alpha);
}
#endif  // custom
