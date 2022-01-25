// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <algorithm>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"
#include "lite/tests/utils/fill_data.h"

#define FP16_MAX_DIFF (5e-1)
#define FP32_ABS_DIFF (1e-7)
#define FP32_RELATIVE_DIFF (1e-4)
#define FP16_ABS_DIFF (1e-1)
#define FP16_RELATIVE_DIFF (1e-1)
namespace paddle {
namespace lite {

template <typename dtype>
void fill_data(dtype* x, const int length, int set_value = -1) {
  if (set_value == -1) {
    for (size_t idx = 0; idx < length; ++idx) {
      x[idx] = idx;
    }
  } else if (set_value != -1) {
    for (size_t idx = 0; idx < length; ++idx) {
      x[idx] = set_value;
    }
  }
}

int randint(int beg, int end) {
  int res = 0;
  fill_data_rand<int>(&res, beg, end, 1);
  return res;
}

bool randbool() { return randint(0, 1000000) < 500000; }

template <class T>
T* AtLogicInd(T* data,
              const std::vector<int>& dim,
              const std::vector<int>& logic_index) {
  assert(dim.size() == logic_index.size());

  int offset = 0;
  int stride = 1;
  for (int i = dim.size() - 1; i >= 0; --i) {
    int ind = logic_index[i];
    if (dim[i] == 1) {
      ind = 0;
    }
    assert(ind < dim[i]);
    offset += ind * stride;
    stride *= dim[i];
  }
  return data + offset;
}

std::vector<int> GenLogicIndex(int logic_offset, const std::vector<int>& dim) {
  std::vector<int> strides(dim.size(), 1);
  for (int i = dim.size() - 2; i >= 0; --i) {
    strides[i] = strides[i + 1] * dim[i + 1];
  }
  std::vector<int> ret(dim.size(), 0);
  for (int i = 0; i < dim.size(); ++i) {
    ret[i] = logic_offset / strides[i];
    logic_offset %= strides[i];
  }
  return ret;
}

template <class T>
void BroadcastCPURef(const T* x,
                     const T* y,
                     T* z,
                     const std::vector<int>& x_dim,
                     const std::vector<int>& y_dim,
                     const std::vector<int>& z_dim,
                     bool use_relu,
                     const std::function<T(T, T)> op) {
  int N = 1;
  for (int i = 0; i < z_dim.size(); ++i) {
    N *= z_dim[i];
  }
  for (int i = 0; i < N; ++i) {
    auto logic_index = GenLogicIndex(i, z_dim);
    const T* x_d = AtLogicInd(x, x_dim, logic_index);
    const T* y_d = AtLogicInd(y, y_dim, logic_index);
    T* z_d = AtLogicInd(z, z_dim, logic_index);
    *z_d = op(*x_d, *y_d);
    if (use_relu) {
      *z_d = std::max(*z_d, static_cast<T>(0));
    }
  }
}

template <class T>
void RunElementwiseBroadcast(const Place& place,
                             const int dim_size,
                             bool fuse_act,
                             const lite_api::CLPrecisionType p,
                             const std::string& alias,
                             const std::string& elt_type,
                             const std::string& act_type,
                             const std::function<T(T, T)> op,
                             double abs_error = 1e-3) {
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  LOG(INFO) << "\n\t[  START  ] Test Precision="
            << lite_api::CLPrecisionTypeToStr(p);
  // set kernel
  auto kernels = KernelRegistry::Global().Create(
      elt_type, TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());

  auto elemul_img_kernel = std::move(kernels.front());
  VLOG(4) << "get elemul kernel: " << elemul_img_kernel->doc();
  const int MAX_SHAPE_VALUE = 10;
  // gen out_dim
  std::vector<int> out_shape(dim_size, 0);
  for (int i = 0; i < dim_size; ++i) {
    out_shape[i] = randint(2, MAX_SHAPE_VALUE);
  }
  std::vector<int> x_shape_full = out_shape;
  std::vector<int> y_shape_full = out_shape;

  std::vector<int> x_shape_cut;
  std::vector<int> y_shape_cut;

  int axis = -1;
  static bool cut_dimension = true;
  cut_dimension = !cut_dimension;
  if (cut_dimension) {
    // generate x_shape_cut and y_shape_cut by remove dimension
    static bool use_axis = true;
    use_axis = !use_axis;
    if (use_axis) {
      x_shape_cut = x_shape_full;
      // we will cut y only, and set tail of y to be 1
      axis = randint(0, dim_size - 1);

      int tail1_num = randint(0, dim_size - axis);
      for (int i = 0; i < axis; ++i) {
        y_shape_full[i] = 1;
      }
      for (int i = axis; i < (dim_size - tail1_num); ++i) {
        y_shape_cut.push_back(y_shape_full[i]);
      }
      for (int i = 0; i < tail1_num; ++i) {
        y_shape_cut.push_back(1);
      }
      for (int i = dim_size - tail1_num; i < dim_size; ++i) {
        y_shape_full[i] = 1;
      }
      static bool swap_x_and_y = true;
      swap_x_and_y = !swap_x_and_y;
      if (swap_x_and_y) {
        std::swap(x_shape_cut, y_shape_cut);
        std::swap(x_shape_full, y_shape_full);
      }
    } else {
      // we will cut x or y
      if (randbool()) {
        y_shape_cut = y_shape_full;
        int cut_x_num = randint(0, dim_size) * randbool();
        for (int i = 0; i < cut_x_num; ++i) {
          x_shape_full[i] = 1;
        }
        for (int i = cut_x_num; i < dim_size; ++i) {
          x_shape_cut.push_back(x_shape_full[i]);
        }
      } else {
        x_shape_cut = x_shape_full;
        int cut_y_num = randint(0, dim_size) * randbool();
        for (int i = 0; i < cut_y_num; ++i) {
          y_shape_full[i] = 1;
        }
        for (int i = cut_y_num; i < dim_size; ++i) {
          y_shape_cut.push_back(y_shape_full[i]);
        }
      }
    }
  } else {
    // generate x_shape_cut and y_shape_cut by random
    // random assign 1 to some dim
    for (int i = 0; i < dim_size; ++i) {
      if (randbool() && y_shape_full[i] != 1) {
        x_shape_full[i] = 1;
      }
      if (randbool() && x_shape_full[i] != 1) {
        y_shape_full[i] = 1;
      }
    }
    // just remove 1 at high dimesion
    int ind = 0;
    while (x_shape_full[ind] == 1) {
      ++ind;
    }
    for (int i = ind; i < dim_size; ++i) {
      x_shape_cut.push_back(x_shape_full[i]);
    }
    ind = 0;
    while (y_shape_full[ind] == 1) {
      ++ind;
    }
    for (int i = ind; i < dim_size; ++i) {
      y_shape_cut.push_back(y_shape_full[i]);
    }
  }

  DDim x_dim =
      DDim(std::vector<int64_t>(x_shape_cut.begin(), x_shape_cut.end()));
  DDim y_dim =
      DDim(std::vector<int64_t>(y_shape_cut.begin(), y_shape_cut.end()));
  DDim out_dim = DDim(std::vector<int64_t>(out_shape.begin(), out_shape.end()));

  LOG(INFO) << "==================" << elt_type << "===================";
  LOG(INFO) << "x_dim:" << x_dim << "\ty_dim:" << y_dim
            << "\tout_dim:" << out_dim;
  LOG(INFO) << "fuse_act:" << fuse_act << "; axis:" << axis;

  // tensor
  lite::Tensor ele_x, ele_y, ele_out;
  ele_x.Resize(x_dim);
  ele_y.Resize(y_dim);
  ele_out.Resize(out_dim);

  // initialize tensors
  VLOG(4) << "initialize tensors";
  paddle::lite::CLImageConverterDefault* default_convertor =
      new CLImageConverterDefault();

  // operator param
  operators::FusionElementwiseActivationParam
      fuseEleParam;  // enabled if fuse_act is true
  fuseEleParam.X = &ele_x;
  fuseEleParam.Y = &ele_y;
  fuseEleParam.Out = &ele_out;
  fuseEleParam.axis = axis;
  fuseEleParam.act_type = fuse_act ? "relu" : "";

  operators::ElementwiseParam eleParam;
  eleParam.X = &ele_x;
  eleParam.Y = &ele_y;
  eleParam.Out = &ele_out;
  eleParam.axis = axis;

  if (fuse_act) {
    elemul_img_kernel->SetParam(fuseEleParam);
  } else {
    elemul_img_kernel->SetParam(eleParam);
  }

  elemul_img_kernel->SetContext(std::move(context));

  // x
  std::vector<float> x_v(x_dim.production());
  // fill_data<float>(x_v.data(), x_v.size());
  fill_data_rand(x_v.data(), -10.f, 10.f, x_dim.production());
  auto x_img_shape = default_convertor->InitImageDimInfoWith(x_dim);  // w, h
  const size_t dtype_size = fp16_flag ? sizeof(half_t) : sizeof(float);
  std::vector<char> x_image_data(x_img_shape.production() * 4 *
                                 dtype_size);  // 4: RGBA
  default_convertor->NCHWToImage(x_v.data(), x_image_data.data(), x_dim);
  MUTABLE_DATA_GPU(&ele_x, x_img_shape[0], x_img_shape[1], x_image_data.data());

  // y
  std::vector<float> y_v(y_dim.production());
  // fill_data<float>(y_v.data(), y_v.size());
  fill_data_rand(y_v.data(), -10.f, 10.f, y_dim.production());
  auto y_img_shape = default_convertor->InitImageDimInfoWith(y_dim);  // w, h
  std::vector<char> y_image_data(y_img_shape.production() * 4 *
                                 dtype_size);  // 4: RGBA
  default_convertor->NCHWToImage(y_v.data(), y_image_data.data(), y_dim);
  MUTABLE_DATA_GPU(&ele_y, y_img_shape[0], y_img_shape[1], y_image_data.data());

  // out
  std::vector<float> out_from_gpu(out_dim.production());
  auto out_img_shape =
      default_convertor->InitImageDimInfoWith(out_dim);  // w, h
  auto* out_image =
      MUTABLE_DATA_GPU(&ele_out, out_img_shape[0], out_img_shape[1], nullptr);

  // run kernel
  elemul_img_kernel->Launch();
  CLRuntime::Global()->command_queue().finish();
  // download gpu result to cpu
  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  std::vector<char> out_image_data(out_img_shape.production() * 4 *
                                   dtype_size);  // 4 : RGBA
  TargetWrapperCL::ImgcpySync(out_image_data.data(),
                              out_image,
                              out_img_shape[0],
                              out_img_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  default_convertor->ImageToNCHW(
      out_image_data.data(), out_from_gpu.data(), out_img_shape, out_dim);

  // compute cpu reference
  std::unique_ptr<float[]> out_from_cpu(new float[out_dim.production()]);
  BroadcastCPURef<float>(x_v.data(),
                         y_v.data(),
                         out_from_cpu.get(),
                         x_shape_full,
                         y_shape_full,
                         out_shape,
                         fuse_act,
                         op);

  VLOG(4) << "output_data vs output_ref_data";
  auto relative_diff_thres =
      fp16_flag ? FP16_RELATIVE_DIFF : FP32_RELATIVE_DIFF;
  auto abs_diff_thres = fp16_flag ? FP16_ABS_DIFF : FP32_ABS_DIFF;
  uint32_t diff_cnt = 0;
  for (int i = 0; i < out_dim.production(); i++) {
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
    LOG(FATAL) << "Err num " << diff_cnt << "/" << out_dim.production();
  }

  LOG(INFO) << "\n\t[  PASSED  ] "
            << " Test Precision=" << lite_api::CLPrecisionTypeToStr(p);
}

template <class T>
void RunElementwiseCommonSize(std::vector<int> x_shape_full,
                              std::vector<int> y_shape_full,
                              std::vector<int> x_shape_cut,
                              std::vector<int> y_shape_cut,
                              std::vector<int> out_shape,
                              int axis,
                              bool xy_swap_flag,
                              double abs_error = 1e-3) {
  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  const lite_api::CLPrecisionType p =
      paddle::lite_api::CLPrecisionType::CL_PRECISION_FP32;
  CLRuntime::Global()->set_precision(p);
  const bool fp16_flag = (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
  LOG(INFO) << "\n\t[  START  ] Test Precision="
            << lite_api::CLPrecisionTypeToStr(p);
  // set kernel
  std::string elt_type = "elementwise_add";
  const std::function<T(T, T)> op = [](float l, float r) { return l + r; };
  bool fuse_act = false;
  auto kernels = KernelRegistry::Global().Create(
      elt_type, TARGET(kOpenCL), PRECISION(kFP16), DATALAYOUT(kImageDefault));
  ASSERT_FALSE(kernels.empty());

  auto elemul_img_kernel = std::move(kernels.front());
  VLOG(4) << "get elemul kernel: " << elemul_img_kernel->doc();

  if (xy_swap_flag) {
    std::swap(x_shape_cut, y_shape_cut);
    std::swap(x_shape_full, y_shape_full);
  }

  DDim x_dim =
      DDim(std::vector<int64_t>(x_shape_cut.begin(), x_shape_cut.end()));
  DDim y_dim =
      DDim(std::vector<int64_t>(y_shape_cut.begin(), y_shape_cut.end()));
  DDim out_dim = DDim(std::vector<int64_t>(out_shape.begin(), out_shape.end()));

  LOG(INFO) << "==================" << elt_type << "===================";
  LOG(INFO) << "x_dim:" << x_dim << "\ty_dim:" << y_dim
            << "\tout_dim:" << out_dim;
  LOG(INFO) << "fuse_act:" << fuse_act << "; axis:" << axis;

  // tensor
  lite::Tensor ele_x, ele_y, ele_out;
  ele_x.Resize(x_dim);
  ele_y.Resize(y_dim);
  ele_out.Resize(out_dim);

  // initialize tensors
  VLOG(4) << "initialize tensors";
  paddle::lite::CLImageConverterDefault* default_convertor =
      new CLImageConverterDefault();

  // operator param
  operators::FusionElementwiseActivationParam
      fuseEleParam;  // enabled if fuse_act is true
  fuseEleParam.X = &ele_x;
  fuseEleParam.Y = &ele_y;
  fuseEleParam.Out = &ele_out;
  fuseEleParam.axis = axis;
  fuseEleParam.act_type = fuse_act ? "relu" : "";

  operators::ElementwiseParam eleParam;
  eleParam.X = &ele_x;
  eleParam.Y = &ele_y;
  eleParam.Out = &ele_out;
  eleParam.axis = axis;

  if (fuse_act) {
    elemul_img_kernel->SetParam(fuseEleParam);
  } else {
    elemul_img_kernel->SetParam(eleParam);
  }

  elemul_img_kernel->SetContext(std::move(context));

  // x
  std::vector<float> x_v(x_dim.production());
  // fill_data<float>(x_v.data(), x_v.size());
  fill_data_rand(x_v.data(), -10.f, 10.f, x_dim.production());
  auto x_img_shape = default_convertor->InitImageDimInfoWith(x_dim);  // w, h
  const size_t dtype_size = fp16_flag ? sizeof(half_t) : sizeof(float);
  std::vector<char> x_image_data(x_img_shape.production() * 4 *
                                 dtype_size);  // 4: RGBA
  default_convertor->NCHWToImage(x_v.data(), x_image_data.data(), x_dim);
  MUTABLE_DATA_GPU(&ele_x, x_img_shape[0], x_img_shape[1], x_image_data.data());

  // y
  std::vector<float> y_v(y_dim.production());
  // fill_data<float>(y_v.data(), y_v.size());
  fill_data_rand(y_v.data(), -10.f, 10.f, y_dim.production());
  auto y_img_shape = default_convertor->InitImageDimInfoWith(y_dim);  // w, h
  std::vector<char> y_image_data(y_img_shape.production() * 4 *
                                 dtype_size);  // 4: RGBA
  default_convertor->NCHWToImage(y_v.data(), y_image_data.data(), y_dim);
  MUTABLE_DATA_GPU(&ele_y, y_img_shape[0], y_img_shape[1], y_image_data.data());

  // out
  std::vector<float> out_from_gpu(out_dim.production());
  auto out_img_shape =
      default_convertor->InitImageDimInfoWith(out_dim);  // w, h
  auto* out_image =
      MUTABLE_DATA_GPU(&ele_out, out_img_shape[0], out_img_shape[1], nullptr);

  // run kernel
  elemul_img_kernel->Launch();
  CLRuntime::Global()->command_queue().finish();
  // download gpu result to cpu
  const size_t cl_image2d_row_pitch{0};
  const size_t cl_image2d_slice_pitch{0};
  std::vector<char> out_image_data(out_img_shape.production() * 4 *
                                   dtype_size);  // 4 : RGBA
  TargetWrapperCL::ImgcpySync(out_image_data.data(),
                              out_image,
                              out_img_shape[0],
                              out_img_shape[1],
                              cl_image2d_row_pitch,
                              cl_image2d_slice_pitch,
                              IoDirection::DtoH);
  default_convertor->ImageToNCHW(
      out_image_data.data(), out_from_gpu.data(), out_img_shape, out_dim);

  // compute cpu reference
  std::unique_ptr<float[]> out_from_cpu(new float[out_dim.production()]);
  BroadcastCPURef<float>(x_v.data(),
                         y_v.data(),
                         out_from_cpu.get(),
                         x_shape_full,
                         y_shape_full,
                         out_shape,
                         fuse_act,
                         op);

  VLOG(4) << "output_data vs output_ref_data";
  auto relative_diff_thres =
      fp16_flag ? FP16_RELATIVE_DIFF : FP32_RELATIVE_DIFF;
  auto abs_diff_thres = fp16_flag ? FP16_ABS_DIFF : FP32_ABS_DIFF;
  uint32_t diff_cnt = 0;
  for (int i = 0; i < out_dim.production(); i++) {
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
      break;
    }
  }
  if (diff_cnt != 0) {
    LOG(FATAL) << "Err num " << diff_cnt << "/" << out_dim.production();
  }

  LOG(INFO) << "\n\t[  PASSED  ] "
            << " Test Precision=" << lite_api::CLPrecisionTypeToStr(p);
}

void test_elementwise_all_dim_data_gpu() {
  // test elementwise common size, only add compute, data in gpu
  int n = 40;
  int c = 40;
  int h = 40;
  int w = 40;
  n = randint(1, 40);
  c = randint(1, 40);
  h = randint(1, 40);
  w = randint(1, 40);
  n = 2;
  c = 3;
  h = 4;
  w = 5;
  std::vector<bool> xy_swap_flags{false, true};
  for (auto xy_swap_flag : xy_swap_flags) {
    RunElementwiseCommonSize<float>({n, c, h, w},
                                    {n, c, h, w},
                                    {n, c, h, w},
                                    {n, c, h, w},
                                    {n, c, h, w},
                                    0,
                                    xy_swap_flag);

    RunElementwiseCommonSize<float>({n, c, h, w},
                                    {1, c, 1, 1},
                                    {n, c, h, w},
                                    {1, c, 1, 1},
                                    {n, c, h, w},
                                    -1,
                                    xy_swap_flag);

    RunElementwiseCommonSize<float>({n, c, h, w},
                                    {1, c, 1, 1},
                                    {n, c, h, w},
                                    {c},
                                    {n, c, h, w},
                                    1,
                                    xy_swap_flag);

    RunElementwiseCommonSize<float>({n, c, h, w},
                                    {1, 1, h, 1},
                                    {n, c, h, w},
                                    {h},
                                    {n, c, h, w},
                                    2,
                                    xy_swap_flag);

    RunElementwiseCommonSize<float>({n, c, h, w},
                                    {1, 1, 1, w},
                                    {n, c, h, w},
                                    {w},
                                    {n, c, h, w},
                                    -1,
                                    xy_swap_flag);

    RunElementwiseCommonSize<float>({n, c, h, w},
                                    {1, c, h, w},
                                    {n, c, h, w},
                                    {c, h, w},
                                    {n, c, h, w},
                                    -1,
                                    xy_swap_flag);

    RunElementwiseCommonSize<float>({n, c, h, w},
                                    {n, c, h, 1},
                                    {n, c, h, w},
                                    {n, c, h},
                                    {n, c, h, w},
                                    0,
                                    xy_swap_flag);

    RunElementwiseCommonSize<float>({n, c, h, w},
                                    {n, c, 1, 1},
                                    {n, c, h, w},
                                    {n, c},
                                    {n, c, h, w},
                                    0,
                                    xy_swap_flag);

    RunElementwiseCommonSize<float>({n, c, h, w},
                                    {1, c, h, 1},
                                    {n, c, h, w},
                                    {c, h},
                                    {n, c, h, w},
                                    1,
                                    xy_swap_flag);

    RunElementwiseCommonSize<float>({n, c, h, w},
                                    {1, 1, h, w},
                                    {n, c, h, w},
                                    {h, w},
                                    {n, c, h, w},
                                    -1,
                                    xy_swap_flag);

    RunElementwiseCommonSize<float>(
        {n, c, h}, {n, c, h}, {n, c, h}, {n, c, h}, {n, c, h}, 0, xy_swap_flag);

    RunElementwiseCommonSize<float>(
        {n, c, h}, {1, 1, h}, {n, c, h}, {h}, {n, c, h}, -1, xy_swap_flag);

    RunElementwiseCommonSize<float>(
        {n, c, h}, {1, c, 1}, {n, c, h}, {c}, {n, c, h}, 1, xy_swap_flag);

    RunElementwiseCommonSize<float>(
        {n, c, h}, {n, 1, 1}, {n, c, h}, {n}, {n, c, h}, 0, xy_swap_flag);

    RunElementwiseCommonSize<float>(
        {n, c, h}, {n, c, 1}, {n, c, h}, {n, c}, {n, c, h}, 0, xy_swap_flag);

    RunElementwiseCommonSize<float>(
        {n, c, h}, {1, c, h}, {n, c, h}, {c, h}, {n, c, h}, -1, xy_swap_flag);

    RunElementwiseCommonSize<float>(
        {h, w}, {h, 1}, {h, w}, {h}, {h, w}, 0, xy_swap_flag);

    RunElementwiseCommonSize<float>(
        {h, w}, {1, w}, {h, w}, {w}, {h, w}, -1, xy_swap_flag);
  }
}

void test_elementwise_broadcast_all_op() {
  const int TEST_RETEAT_NUM = 1;
  std::vector<bool> relu_flag_v{false, true};
  for (int repeat_count = 0; repeat_count < TEST_RETEAT_NUM; ++repeat_count) {
    for (int dim_size = 4; dim_size <= 4; dim_size++) {
      for (auto fuse_act : relu_flag_v) {
        for (const auto precision_type :
             {paddle::lite_api::CLPrecisionType::CL_PRECISION_FP32}) {
          RunElementwiseBroadcast<float>(
              TARGET(kOpenCL),
              dim_size,
              fuse_act,
              precision_type,
              "def",
              "elementwise_add",
              "",
              [](float l, float r) { return l + r; });
          RunElementwiseBroadcast<float>(
              TARGET(kOpenCL),
              dim_size,
              true,
              precision_type,
              "def",
              "fusion_elementwise_add_activation",
              "",
              [](float l, float r) { return l + r; });
          RunElementwiseBroadcast<float>(
              TARGET(kOpenCL),
              dim_size,
              fuse_act,
              precision_type,
              "def",
              "elementwise_sub",
              "",
              [](float l, float r) { return l - r; });
          RunElementwiseBroadcast<float>(
              TARGET(kOpenCL),
              dim_size,
              true,
              precision_type,
              "def",
              "fusion_elementwise_sub_activation",
              "",
              [](float l, float r) { return l - r; });
          RunElementwiseBroadcast<float>(
              TARGET(kOpenCL),
              dim_size,
              fuse_act,
              precision_type,
              "def",
              "elementwise_mul",
              "",
              [](float l, float r) { return l * r; });
          RunElementwiseBroadcast<float>(
              TARGET(kOpenCL),
              dim_size,
              true,
              precision_type,
              "def",
              "fusion_elementwise_mul_activation",
              "",
              [](float l, float r) { return l * r; });
          RunElementwiseBroadcast<float>(
              TARGET(kOpenCL),
              dim_size,
              fuse_act,
              precision_type,
              "def",
              "elementwise_div",
              "",
              [](float l, float r) { return l / r; });
          RunElementwiseBroadcast<float>(
              TARGET(kOpenCL),
              dim_size,
              true,
              precision_type,
              "def",
              "fusion_elementwise_div_activation",
              "",
              [](float l, float r) { return l / r; });
          RunElementwiseBroadcast<float>(
              TARGET(kOpenCL),
              dim_size,
              fuse_act,
              precision_type,
              "def",
              "elementwise_max",
              "",
              [](float l, float r) { return fmax(l, r); });
          RunElementwiseBroadcast<float>(
              TARGET(kOpenCL),
              dim_size,
              fuse_act,
              precision_type,
              "def",
              "elementwise_min",
              "",
              [](float l, float r) { return fmin(l, r); });
          RunElementwiseBroadcast<float>(
              TARGET(kOpenCL),
              dim_size,
              fuse_act,
              precision_type,
              "def",
              "elementwise_pow",
              "",
              [](float l, float r) { return pow(l, r); });
          RunElementwiseBroadcast<float>(
              TARGET(kOpenCL),
              dim_size,
              fuse_act,
              precision_type,
              "def",
              "elementwise_mod",
              "",
              [](float l, float r) { return fmod(l, r); });
        }
      }
    }
  }
}

TEST(elementwise_broadcast, compute_basic) {
  // test elementwise broadcast
  test_elementwise_broadcast_all_op();

  // test elementwise all dims, only add compute, data in gpu
  test_elementwise_all_dim_data_gpu();
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(elementwise_div, kOpenCL, kFP16, kImageDefault, def);
USE_LITE_KERNEL(elementwise_add, kOpenCL, kFP16, kImageDefault, def);
USE_LITE_KERNEL(elementwise_sub, kOpenCL, kFP16, kImageDefault, def);
USE_LITE_KERNEL(elementwise_mul, kOpenCL, kFP16, kImageDefault, def);
USE_LITE_KERNEL(elementwise_max, kOpenCL, kFP16, kImageDefault, def);
USE_LITE_KERNEL(elementwise_min, kOpenCL, kFP16, kImageDefault, def);
USE_LITE_KERNEL(elementwise_pow, kOpenCL, kFP16, kImageDefault, def);
USE_LITE_KERNEL(elementwise_mod, kOpenCL, kFP16, kImageDefault, def);
USE_LITE_KERNEL(
    fusion_elementwise_add_activation, kOpenCL, kFP16, kImageDefault, def);
USE_LITE_KERNEL(
    fusion_elementwise_sub_activation, kOpenCL, kFP16, kImageDefault, def);
USE_LITE_KERNEL(
    fusion_elementwise_mul_activation, kOpenCL, kFP16, kImageDefault, def);
USE_LITE_KERNEL(
    fusion_elementwise_div_activation, kOpenCL, kFP16, kImageDefault, def);
