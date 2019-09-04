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

#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <memory>
#include <random>
#include <vector>
#include "lite/backends/opencl/cl_context.h"
#include "lite/backends/opencl/cl_runtime.h"
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/tensor.h"
#include "lite/utils/cp_logging.h"

DEFINE_string(cl_path, "/data/local/tmp/opencl", "The OpenCL kernels path.");

namespace paddle {
namespace lite {

template <typename Dtype>
void PrintData(std::string name, Dtype *a, const int rows, const int cols) {
  std::cout << "==== " << name << " ====" << std::endl;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      std::cout << " " << a[r * cols + c];
    }
    std::cout << std::endl;
  }
}

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
  return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template <typename Dtype>
void im2col(const Dtype *data_im,
            const int channels,
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
            Dtype *data_col) {
  const int output_h =
      (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
  const int output_w =
      (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
  const int channel_size = height * width;

  for (int channel = 0; channel++ < channels; data_im += channel_size) {
    for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
      for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
        int input_row = -pad_h + kernel_row * dilation_h;
        for (int output_rows = 0; output_rows < output_h; ++output_rows) {
          if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
            for (int output_cols = 0; output_cols < output_w; ++output_cols) {
              *(data_col++) = 0;
            }
          } else {
            int input_col = -pad_w + kernel_col * dilation_w;
            for (int output_col = 0; output_col < output_w; ++output_col) {
              *(data_col++) = (is_a_ge_zero_and_a_lt_b(input_col, width))
                                  ? data_im[input_row * width + input_col]
                                  : 0;
              input_col += stride_w;
            }
          }
          input_row += stride_h;
        }
      }
    }
  }
}

// #define CHECK_ERROR
// #define PRINT_RESULT
// #define LOOP_TEST
TEST(cl_test, im2col_test) {
  using T = float;
  std::string kernel_func_name = "im2col";
  std::string kernel_func_path = "buffer/im2col_kernel.cl";

#ifdef LOOP_TEST
  for (int n : {1}) {
    for (int c : {32}) {
      for (int h : {224}) {
        for (int w : {224}) {
          for (int kernel_h : {3}) {
            for (int kernel_w : {3}) {
              for (int pad_h : {1}) {
                for (int pad_w : {1}) {
                  for (int stride_h : {2}) {
                    for (int stride_w : {2}) {
                      for (int dilation_h : {1}) {
                        for (int dilation_w : {1}) {
// TODO(yuanshuai): support group for im2col
#else
  int n = 8;
  int c = 32;
  int h = 224;
  int w = 224;
  int kernel_h = 3;
  int kernel_w = 3;
  int pad_h = 1;
  int pad_w = 1;
  int stride_h = 2;
  int stride_w = 2;
  int dilation_h = 1;
  int dilation_w = 1;
#endif

                          int img_offset = 0;
                          int col_offset = 0;

                          std::vector<DDim::value_type> input_shape{n, c, h, w};
                          int channels = input_shape[1];
                          int height = input_shape[2];
                          int width = input_shape[3];

                          int height_col = (height + 2 * pad_h -
                                            (dilation_h * (kernel_h - 1) + 1)) /
                                               stride_h +
                                           1;
                          int width_col = (width + 2 * pad_w -
                                           (dilation_w * (kernel_w - 1) + 1)) /
                                              stride_w +
                                          1;
                          int col_chw = channels * kernel_h * kernel_w *
                                        height_col * width_col;
                          if (col_chw <= 0 || height_col <= 0 ||
                              width_col <= 0 || channels <= 0) {
                            VLOG(4) << "col_chw <= 0, skipped";
#ifdef LOOP_TEST
                            continue;
#else
                            return;
#endif
                          }

                          VLOG(4) << "kernel_func_name:" << kernel_func_name
                                  << " kernel_func_path:" << kernel_func_path;
                          VLOG(4) << "input_shape:" << input_shape[0] << ", "
                                  << input_shape[1] << ", " << input_shape[2]
                                  << ", " << input_shape[3];
                          VLOG(4) << "kernel_h:" << kernel_h
                                  << " kernel_w:" << kernel_w
                                  << " pad_h:" << pad_h << " pad_w:" << pad_w
                                  << " stride_h:" << stride_h
                                  << " stride_w:" << stride_w
                                  << " dilation_h:" << dilation_h
                                  << " dilation_w:" << dilation_w;
                          VLOG(4) << "height_col:" << height_col
                                  << " width_col:" << width_col
                                  << " img_offset:" << img_offset
                                  << " col_offset:" << col_offset
                                  << " col_chw:" << col_chw;

                          const DDim input_dim = DDim(input_shape);
                          const int input_elem_num = input_dim.production();
                          T *in_data = static_cast<T *>(
                              calloc(sizeof(T), input_elem_num));
                          T *out_data =
                              static_cast<T *>(calloc(sizeof(T), n * col_chw));
                          T *out_ref_data =
                              static_cast<T *>(calloc(sizeof(T), n * col_chw));
                          for (int i = 0; i < input_elem_num; ++i) {
                            in_data[i] = i;
                          }

                          // CPU im2col
                          for (int b = 0; b < n; b++) {
                            im2col<T>(in_data + b * channels * height * width,
                                      channels,
                                      height,
                                      width,
                                      kernel_h,
                                      kernel_w,
                                      pad_h,
                                      pad_w,
                                      stride_h,
                                      stride_w,
                                      dilation_h,
                                      dilation_w,
                                      out_ref_data + b * col_chw);
                          }

                          // OpenCL im2col
                          auto *runtime = CLRuntime::Global();
                          CHECK(runtime->IsInitSuccess())
                              << "Fail to initialize OpenCL runtime.";
                          runtime->set_cl_path(FLAGS_cl_path);

                          std::unique_ptr<CLContext> context(new CLContext);
                          context->AddKernel(kernel_func_name,
                                             kernel_func_path);
                          auto kernel = context->GetKernel(kernel_func_name);

                          auto *d_in =
                              static_cast<cl::Buffer *>(TargetWrapperCL::Malloc(
                                  sizeof(T) * input_elem_num));
                          auto *d_out = static_cast<cl::Buffer *>(
                              TargetWrapperCL::Malloc(sizeof(T) * n * col_chw));
                          TargetWrapperCL::MemcpySync(
                              d_in,
                              in_data,
                              sizeof(T) * input_elem_num,
                              IoDirection::HtoD);

                          int n_threads = channels * height_col * width_col;
                          cl_int status;
                          int arg_idx = 0;
                          for (int b = 0; b < n; b++) {
                            img_offset = b * channels * height * width;
                            col_offset = b * col_chw;
                            arg_idx = 0;
                            status = kernel.setArg(arg_idx, *d_in);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, img_offset);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, n_threads);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, height);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, width);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, kernel_h);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, kernel_w);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, pad_h);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, pad_w);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, stride_h);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, stride_w);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, dilation_h);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, dilation_w);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, height_col);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, width_col);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, *d_out);
                            CL_CHECK_FATAL(status);
                            status = kernel.setArg(++arg_idx, col_offset);
                            CL_CHECK_FATAL(status);

                            auto global_work_size =
                                cl::NDRange{static_cast<size_t>(col_chw)};
                            status =
                                context->GetCommandQueue().enqueueNDRangeKernel(
                                    kernel,
                                    cl::NullRange,
                                    global_work_size,
                                    cl::NullRange,
                                    nullptr,
                                    nullptr);
                            CL_CHECK_FATAL(status);
                          }

                          status = context->GetCommandQueue().finish();
                          CL_CHECK_FATAL(status);

                          TargetWrapperCL::MemcpySync(out_data,
                                                      d_out,
                                                      sizeof(T) * n * col_chw,
                                                      IoDirection::DtoH);

#ifdef PRINT_RESULT
                          PrintData("in", in_data, height, width);
                          PrintData("out_ref", out_ref_data, height, width);
                          PrintData("out", out_data, height, width);
#endif

                          for (int i = 0; i < n * col_chw; ++i) {
                            EXPECT_NEAR(out_data[i], out_ref_data[i], 1e-5);
#ifdef CHECK_ERROR
                            if (abs(out_data[i] - out_ref_data[i]) > 1e-5) {
                              std::cout << "i:" << i << std::endl;
                              PrintData("in", in_data, height, width);
                              PrintData("out_ref", out_ref_data, height, width);
                              PrintData("out", out_data, height, width);
                              exit(0);
                            }
#endif
                          }

                          free(in_data);
                          free(out_data);
                          free(out_ref_data);
                          TargetWrapperCL::Free(d_in);
                          TargetWrapperCL::Free(d_out);

#ifdef LOOP_TEST
                        }  // dilation_w
                      }    // dilation_h
                    }      // stride_w
                  }        // stride_h
                }          // pad_w
              }            // pad_h
            }              // kernel_w
          }                // kernel_h
        }                  // w
      }                    // h
    }                      // c
  }                        // n
#endif
}

}  // namespace lite
}  // namespace paddle
