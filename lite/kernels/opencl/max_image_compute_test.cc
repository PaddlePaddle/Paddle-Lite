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

#include <gtest/gtest.h>
#include <memory>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"
#include "lite/tests/utils/fill_data.h"

#define FP16_MAX_DIFF (5e-1)
#define FP32_ABS_DIFF (1e-7)
#define FP32_RELATIVE_DIFF (1e-6)
#define FP16_ABS_DIFF (1e-3)
#define FP16_RELATIVE_DIFF (1e-3)
namespace paddle {
namespace lite {

void max_n(const float* src,
           float* dst,
           int num_in,
           int channel_in,
           int height_in,
           int width_in) {
  int hw_size = height_in * width_in;
  int chw_size = channel_in * hw_size;
  int data_index, src_index;
  for (int c = 0; c < channel_in; ++c) {
    for (int h = 0; h < height_in; ++h) {
      for (int w = 0; w < width_in; ++w) {
        data_index = c * hw_size + h * width_in + w;
        dst[data_index] = src[data_index];
        for (int n = 1; n < num_in; ++n) {
          src_index = n * chw_size + data_index;
          dst[data_index] = dst[data_index] > src[src_index] ? dst[data_index]
                                                             : src[src_index];
        }
      }
    }
  }
}

void max_c(const float* src,
           float* dst,
           int num_in,
           int channel_in,
           int height_in,
           int width_in) {
  int hw_size = height_in * width_in;
  int chw_size = hw_size * channel_in;
  int data_index, src_index0, src_index;
  for (int n = 0; n < num_in; ++n) {
    for (int h = 0; h < height_in; ++h) {
      for (int w = 0; w < width_in; ++w) {
        data_index = n * hw_size + h * width_in + w;
        src_index0 = n * chw_size + h * width_in + w;
        dst[data_index] = src[src_index0];
        for (int c = 1; c < channel_in; ++c) {
          src_index = src_index0 + c * hw_size;
          dst[data_index] = dst[data_index] > src[src_index] ? dst[data_index]
                                                             : src[src_index];
        }
      }
    }
  }
}

void max_h(const float* src,
           float* dst,
           int num_in,
           int channel_in,
           int height_in,
           int width_in) {
  int cw_size = channel_in * width_in;
  int chw_size = cw_size * height_in;
  int hw_size = height_in * width_in;
  int data_index, src_index, src_index0;
  for (int n = 0; n < num_in; ++n) {
    for (int c = 0; c < channel_in; ++c) {
      for (int w = 0; w < width_in; ++w) {
        data_index = n * cw_size + c * width_in + w;
        src_index0 = n * chw_size + c * hw_size + w;
        dst[data_index] = src[src_index0];
        for (int h = 1; h < height_in; ++h) {
          src_index = src_index0 + h * width_in;
          dst[data_index] = dst[data_index] > src[src_index] ? dst[data_index]
                                                             : src[src_index];
        }
      }
    }
  }
}

void max_w(const float* src,
           float* dst,
           int num_in,
           int channel_in,
           int height_in,
           int width_in) {
  int ch_size = channel_in * height_in;
  int hw_size = height_in * width_in;
  int chw_size = ch_size * width_in;
  int data_index, src_index0, src_index;
  for (int n = 0; n < num_in; ++n) {
    for (int c = 0; c < channel_in; ++c) {
      for (int h = 0; h < height_in; ++h) {
        data_index = n * ch_size + c * height_in + h;
        src_index0 = n * chw_size + c * hw_size + h * width_in;
        dst[data_index] = src[src_index0];
        for (int w = 1; w < width_in; ++w) {
          src_index = src_index0 + w;
          dst[data_index] = dst[data_index] > src[src_index] ? dst[data_index]
                                                             : src[src_index];
        }
      }
    }
  }
}

void max_all(const float* src,
             float* dst,
             int num_in,
             int channel_in,
             int height_in,
             int width_in) {
  float max = src[0];
  int src_index;
  int n_id, c_id;
  for (int n = 0; n < num_in; ++n) {
    n_id = n * channel_in * height_in * width_in;
    for (int c = 0; c < channel_in; ++c) {
      c_id = c * height_in * width_in;
      for (int h = 0; h < height_in; ++h) {
        for (int w = 0; w < width_in; ++w) {
          src_index = n_id + c_id + h * width_in + w;
          max = src[src_index] > max ? src[src_index] : max;
        }
      }
    }
  }
  dst[0] = max;
}

void max_nc(const float* src,
            float* dst,
            int num_in,
            int channel_in,
            int height_in,
            int width_in) {
  // reduce n first.
  DDimLite ddimA({1, channel_in, height_in, width_in});
  lite::Tensor tensor_tmp;
  tensor_tmp.Resize(ddimA);
  float* tmp_out = tensor_tmp.mutable_data<float>();
  max_n(src, tmp_out, num_in, channel_in, height_in, width_in);
  max_c(tmp_out, dst, 1, channel_in, height_in, width_in);
}

void max_ch(const float* src,
            float* dst,
            int num_in,
            int channel_in,
            int height_in,
            int width_in) {
  // reduce c first
  DDimLite ddimA({num_in, 1, height_in, width_in});
  lite::Tensor tensor_tmp;
  tensor_tmp.Resize(ddimA);
  float* tmp_out = tensor_tmp.mutable_data<float>();
  max_c(src, tmp_out, num_in, channel_in, height_in, width_in);
  max_h(tmp_out, dst, num_in, 1, height_in, width_in);
}

void max_hw(const float* src,
            float* dst,
            int num_in,
            int channel_in,
            int height_in,
            int width_in) {
  // reduce h first
  DDimLite ddimA({num_in, channel_in, 1, width_in});
  lite::Tensor tensor_tmp;
  tensor_tmp.Resize(ddimA);
  float* tmp_out = tensor_tmp.mutable_data<float>();
  max_h(src, tmp_out, num_in, channel_in, height_in, width_in);
  max_w(tmp_out, dst, num_in, channel_in, 1, width_in);
}

void max_test(const lite_api::CLPrecisionType p,
              std::vector<int> dim,
              bool keepdims,
              const int n,
              const int c,
              const int h,
              const int w) {
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  LOG(INFO) << "\n\t[  START  ] Test Precision="
            << lite_api::CLPrecisionTypeToStr(p) << " n=" << n << " c=" << c
            << " h=" << h << " w=" << w;

  auto kernels = KernelRegistry::Global().Create(
      "max", TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());

  DDim x_dims = DDim(std::vector<DDim::value_type>{n, c, h, w});
  // bool keep_dim = true;
  bool reduce_all = false;
  auto x_rank = x_dims.size();
  lite::Tensor x, out;
  x.Resize(x_dims);
  if (!dim.empty()) {
    for (size_t i = 0; i < dim.size(); i++) {
      if (dim[i] < 0) {
        dim[i] += x_rank;
      }
    }
  }

  std::stable_sort(dim.begin(), dim.end());
  if (dim.size() == 0) {
    reduce_all = true;
  }

  std::vector<int64_t> out_dims_shape;
  // DDim out_dims;
  if (reduce_all) {
    if (keepdims) {
      for (size_t i = 0; i < x_dims.size(); i++) {
        out_dims_shape.push_back(1);
      }
    } else {
      out_dims_shape.push_back(1);
    }
  } else {
    for (size_t i = 0; i < x_dims.size(); i++) {
      out_dims_shape.push_back(x_dims[i]);
    }
    if (keepdims) {
      for (size_t i = 0; i < dim.size(); ++i) {
        out_dims_shape[dim[i]] = 1L;
      }
    } else {
      int64_t kDelFlag = -2;
      for (size_t i = 0; i < dim.size(); ++i) {
        out_dims_shape[dim[i]] = kDelFlag;
      }
      out_dims_shape.erase(
          remove(out_dims_shape.begin(), out_dims_shape.end(), kDelFlag),
          out_dims_shape.end());
    }
    if (!keepdims && out_dims_shape.empty()) {
      out_dims_shape.push_back(1);
    }
    out.Resize(DDim(out_dims_shape));
  }

  DDim out_dims = DDim(out_dims_shape);
  operators::ReduceParam param;
  param.X = &x;
  param.Out = &out;
  param.dim = dim;
  param.keep_dim = keepdims;
  param.reduce_all = reduce_all;

  kernel->SetParam(param);
  kernel->SetContext(std::move(context));

  std::vector<float> x_cpu(x_dims.production());
  std::vector<float> out_from_cpu(out_dims.production());
  std::vector<float> out_from_gpu(out_dims.production());
  fill_data_rand(x_cpu.data(), -1.f, 1.f, x_dims.production());

  for (int i = 0; i < x_dims.production(); i++) {
    x_cpu[i] = (static_cast<float>i;
    if (i == 2) {
      x_cpu[i] = static_cast<float> 128;
    }
  }

  CLImageConverterDefault* default_converter = new CLImageConverterDefault();
  DDim x_image_shape = default_converter->InitImageDimInfoWith(x_dims);
  DDim out_image_shape = default_converter->InitImageDimInfoWith(out_dims);
  VLOG(4) << "x_image_shape = " << x_image_shape[0] << " " << x_image_shape[1];
  VLOG(4) << "out_image_shape = " << out_image_shape[0] << " "
          << out_image_shape[1];

  const size_t dtype_size = fp16_flag ? sizeof(half_t) : sizeof(float);
  std::vector<char> x_image_data(x_image_shape.production() * 4 * dtype_size);
  default_converter->NCHWToImage(x_cpu.data(), x_image_data.data(), x_dims);
  MUTABLE_DATA_GPU(&x, x_image_shape[0], x_image_shape[1], x_image_data.data());
  auto* out_image =
      MUTABLE_DATA_GPU(&out, out_image_shape[0], out_image_shape[1], nullptr);

  // run opencl kernel
  kernel->Launch();
  CLRuntime::Global()->command_queue().finish();

  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  std::vector<char> out_image_data(out_image_shape.production() * 4 *
                                   dtype_size);  // 4 : RGBA
  TargetWrapperCL::ImgcpySync(out_image_data.data(),
                              out_image,
                              out_image_shape[0],
                              out_image_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  default_converter->ImageToNCHW(
      out_image_data.data(), out_from_gpu.data(), out_image_shape, out_dims);

  // run cpu ref
  if (x_dims.size() == 4) {
    int in_n = x_dims[0];
    int in_c = x_dims[1];
    int in_h = x_dims[2];
    int in_w = x_dims[3];
    if (dim.size() == 0) {
      max_all(x_cpu.data(), out_from_cpu.data(), in_n, in_c, in_h, in_w);
    } else if (dim.size() == 1) {
      switch (dim[0]) {
        case 0:
          max_n(x_cpu.data(), out_from_cpu.data(), in_n, in_c, in_h, in_w);
          break;
        case 1:
          max_c(x_cpu.data(), out_from_cpu.data(), in_n, in_c, in_h, in_w);
          break;
        case 2:
          max_h(x_cpu.data(), out_from_cpu.data(), in_n, in_c, in_h, in_w);
          break;
        case 3:
          max_w(x_cpu.data(), out_from_cpu.data(), in_n, in_c, in_h, in_w);
          break;
        default:
          LOG(FATAL) << "error!!!";
      }
    } else if (dim.size() == 2) {
      if (dim[0] == 0 && dim[1] == 1) {
        max_nc(x_cpu.data(), out_from_cpu.data(), in_n, in_c, in_h, in_w);
      } else if (dim[0] == 1 && dim[1] == 2) {
        max_ch(x_cpu.data(), out_from_cpu.data(), in_n, in_c, in_h, in_w);
      } else if (dim[0] == 2 && dim[1] == 3) {
        max_hw(x_cpu.data(), out_from_cpu.data(), in_n, in_c, in_h, in_w);
      } else {
        LOG(FATAL) << "invalid dim!!";
      }
    } else {
      LOG(FATAL) << "invalid dims_!!";
    }
  }

  VLOG(4) << "output_data vs output_ref_data";
  auto relative_diff_thres =
      fp16_flag ? FP16_RELATIVE_DIFF : FP32_RELATIVE_DIFF;
  auto abs_diff_thres = fp16_flag ? FP16_ABS_DIFF : FP32_ABS_DIFF;
  uint32_t diff_cnt = 0;
  for (int i = 0; i < out_dims.production(); i++) {
    auto relative_diff =
        COMPUTE_RELATIVE_DIFF(out_from_gpu[i], out_from_cpu[i]);
    auto abs_diff = COMPUTE_ABS_DIFF(out_from_gpu[i], out_from_cpu[i]);
    EXPECT_FALSE(relative_diff > relative_diff_thres &&
                 abs_diff > abs_diff_thres);
    if (relative_diff > relative_diff_thres && abs_diff > abs_diff_thres) {
      LOG(WARNING) << lite_api::CLPrecisionTypeToStr(p) << "   err idx: " << i
                   << " abs_diff: " << abs_diff
                   << "\t relative_diff: " << relative_diff
                   << "\t out_ins: " << out_from_gpu[i]
                   << "\t out_ref: " << out_from_cpu[i];
      diff_cnt++;
    }
  }
  if (diff_cnt != 0) {
    LOG(FATAL) << "Err num " << diff_cnt << "/" << out_dims.production();
  }

  LOG(INFO) << "\n\t[  PASSED  ] "
            << " Test Precision=" << lite_api::CLPrecisionTypeToStr(p)
            << " x_dim=" << x_dims;
}

TEST(max, compute_basic) {
  for (const auto precision_type :
       {lite_api::CLPrecisionType::CL_PRECISION_FP32}) {
    std::vector<std::vector<int>> reduce_dim{
        {}, {0}, {1}, {2}, {3}, {0, 1}, {1, 2}, {2, 3}, {-2, -1}};
    for (int n : {1, 10}) {
      for (int c : {1, 29}) {
        for (int h : {1, 20}) {
          for (int w : {1, 30}) {
            for (auto dim : reduce_dim) {
              auto x_dims = DDim(std::vector<int64_t>({n, c, h, w}));
              max_test(precision_type, dim, true, n, c, h, w);
            }
          }
        }
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(max, kOpenCL, kFP16, kImageDefault, image2d);
