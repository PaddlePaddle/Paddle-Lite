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

#include <vector>
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"
#include "lite/utils/string.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class MatMulV2ImageCompute : public KernelLite<TARGET(kOpenCL),
                                               PRECISION(kFP16),
                                               DATALAYOUT(kImageFolder)> {
 public:
  using param_t = operators::MatMulParam;

  void transpose_cpu(const float* in_data,
                     float* out_data,
                     const int in_rows,
                     const int in_cols) {
    CHECK(in_data && out_data && in_rows > 0 && in_cols > 0);
    for (int r = 0; r < in_rows; ++r) {
      for (int c = 0; c < in_cols; ++c) {
        out_data[c * in_rows + r] = in_data[r * in_cols + c];
      }
    }
  }
  //  |  0  1  2  3 16 17 18 19 |
  //  |  4  5  6  7 20 21 22 23 |
  //  |  8  9 10 11 24 25 25 27 |
  //  | 12 13 14 15 28 29 30 31 |
  void RearrangeByBlk4x4(const float* src, void* dst, size_t O, size_t I) {
    bool fp16_support =
        CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;
    VLOG(4) << "fp16_support = " << fp16_support;
    float* dst_fp32 = static_cast<float*>(dst);
    half_t* dst_fp16 = static_cast<half_t*>(dst);

    size_t i_blocks = UP_DIV(I, 4);
    size_t o_blocks = UP_DIV(O, 4);
    size_t dst_index = 0;
    for (size_t block_y = 0; block_y < o_blocks; block_y++) {
      for (size_t block_x = 0; block_x < i_blocks; block_x++) {
        for (size_t y_in_block = 0; y_in_block < 4; y_in_block++) {
          const int y = block_y * 4 + y_in_block;
          for (size_t x_in_block = 0; x_in_block < 4; x_in_block++) {
            const int x = block_x * 4 + x_in_block;
            if (y < O && x < I) {
              fp16_support ? dst_fp16[dst_index++] = Float2Half(src[y * I + x])
                           : dst_fp32[dst_index++] = src[y * I + x];
            } else {
              fp16_support ? dst_fp16[dst_index++] = Float2Half(0.f)
                           : dst_fp32[dst_index++] = 0.f;
            }
          }
        }
      }
    }
  }

  void PrepareForRun() override {
    matmul_v2_param_ = param_.get_mutable<param_t>();
    transpose_x_ = matmul_v2_param_->transpose_X;
    transpose_y_ = matmul_v2_param_->transpose_Y;
    alpha_ = matmul_v2_param_->alpha;
    auto x_dims = matmul_v2_param_->X->dims();
    auto y_t = matmul_v2_param_->Y;
    auto y_dims = y_t->dims();
    if (x_dims.size() == 1 && x_dims == y_dims) {  // for matmul_v2 dim_1x1 case
      transpose_x_ = false;
      transpose_y_ = false;
    }

    const int thres_k = 1024;
    bool precision_forced_to_fp32 = false;
    const bool enable_fp16 =
        CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;
    if (enable_fp16) {
      k_ = transpose_y_ ? y_dims[1] : y_dims[0];
      if (k_ > thres_k) {
        CLRuntime::Global()->set_precision(lite_api::CL_PRECISION_FP32);
        build_options_ += " -DCL_DTYPE_half -DCL_DTYPE_FLOAT_FORCE ";
        precision_forced_to_fp32 = true;
      }
    }

    int k_y = y_dims.size() >= 2 ? y_dims[y_dims.size() - 2] : y_dims[0];
    n_ = y_dims.size() >= 2 ? y_dims[y_dims.size() - 1] : y_dims[0];
    Tensor y_trans_cpu_t;
    VLOG(4) << "persistableY: " << y_t->persistable()
            << ", transposeY: " << transpose_y_;
    if (y_t->persistable()) {
      if (transpose_y_ && y_dims.size() >= 2) {
        y_trans_cpu_t.Resize(y_t->dims());
        if (y_dims.size() == 2) {
          transpose_cpu(y_t->data<float>(),
                        y_trans_cpu_t.mutable_data<float>(),
                        y_t->dims()[0],
                        y_t->dims()[1]);
          y_t = &y_trans_cpu_t;
          k_y = y_dims[1];
          n_ = y_dims[0];
        } else {
          // y_dims.size() > 2
          batch_ = y_dims.count(0, y_dims.size() - 2);
          int y_inner = y_dims[y_dims.size() - 2] * y_dims[y_dims.size() - 1];
          for (int i = 0; i < batch_; ++i) {
            transpose_cpu(y_t->data<float>() + i * y_inner,
                          y_trans_cpu_t.mutable_data<float>() + i * y_inner,
                          y_dims[y_dims.size() - 2],
                          y_dims[y_dims.size() - 1]);
          }
          k_y = y_dims[y_dims.size() - 1];
          n_ = y_dims[y_dims.size() - 2];
        }
        y_t = &y_trans_cpu_t;
      }

      auto y_ext_dims = y_dims;
      if (x_dims.size() == 2 && y_dims.size() == 2) {
        y_ext_dims[0] = ROUND_UP(k_y, 4);
        y_ext_dims[1] = ROUND_UP(n_, 4);
      } else if (x_dims.size() == 1 && y_dims.size() == 1) {
        y_ext_dims = DDim(std::vector<DDim::value_type>{1, 1});
        if (transpose_y_) {
          y_ext_dims[0] = ROUND_UP(1, 4);
          y_ext_dims[1] = ROUND_UP(y_dims[0], 4);
          n_ = y_dims[0], k_y = 1;
        } else {
          y_ext_dims[0] = ROUND_UP(y_dims[0], 4);
          y_ext_dims[1] = ROUND_UP(1, 4);
          n_ = 1, k_y = y_dims[0];
        }
      } else if (y_dims.size() > 2) {
        y_ext_dims[y_dims.size() - 2] = k_y;
        y_ext_dims[y_dims.size() - 1] = n_;
        y_ext_dims[y_dims.size() - 3] = ROUND_UP(y_dims[y_dims.size() - 3], 4);
      } else if (x_dims.size() > 2 && y_dims.size() <= 2) {
        y_ext_dims =
            y_dims.size() == 1
                ? DDim(std::vector<DDim::value_type>{1, 4, 1, y_dims[0]})
                : DDim(std::vector<DDim::value_type>{1, 4, k_y, n_});
      } else if (x_dims.size() == 2 && y_dims.size() == 1) {
        y_ext_dims =
            DDim(std::vector<DDim::value_type>{ROUND_UP(y_dims[0], 4)});
      }

      auto y_cpu_t = std::unique_ptr<Tensor>(new Tensor);
      y_cpu_t->Resize(y_ext_dims);
      auto* y_buffer_data = MUTABLE_DATA_CPU(y_cpu_t.get());
      auto* y_cpu = y_t->data<float>();
      if (x_dims.size() > 2 && y_dims.size() > 2) {
        batch_ = y_dims.count(0, y_dims.size() - 2);
        convert(y_cpu, y_buffer_data, y_ext_dims);
        DDim tmp_dim = y_ext_dims;
        tmp_dim[tmp_dim.size() - 3] = y_dims[y_dims.size() - 3];
        convert(y_cpu, y_buffer_data, tmp_dim);
      } else if (x_dims.size() > 2 && y_dims.size() <= 2) {
        batch_ = x_dims.count(0, x_dims.size() - y_dims.size());
        DDim tmp_dim =
            y_dims.size() == 1
                ? DDim(std::vector<DDim::value_type>{1, 1, 1, y_dims[0]})
                : DDim(std::vector<DDim::value_type>{1, 1, k_y, n_});
        convert(y_cpu, y_buffer_data, tmp_dim);
      } else if (x_dims.size() == 2 && y_dims.size() == 1) {
        batch_ = x_dims.count(0, x_dims.size() - y_dims.size());
        DDim tmp_dim =
            y_dims.size() == 1
                ? DDim(std::vector<DDim::value_type>{1, 1, 1, y_dims[0]})
                : DDim(std::vector<DDim::value_type>{1, 1, k_y, n_});
        float* image_fp32 = static_cast<float*>(y_buffer_data);
        half_t* image_fp16 = static_cast<half_t*>(y_buffer_data);
        bool fp16_support =
            CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;
        for (int i = 0; i < tmp_dim.production(); i++) {
          fp16_support ? image_fp16[i] = Float2Half(y_cpu[i]) : image_fp32[i] =
                                                                    y_cpu[i];
        }
      } else {
        VLOG(4) << "y_ext_dims: " << y_ext_dims;
        RearrangeByBlk4x4(y_cpu, y_buffer_data, k_y, n_);
      }

      auto& context = ctx_->As<OpenCLContext>();
      CHECK(context.cl_context() != nullptr);
      is_mali_ = context.cl_context()->IsArmMali();
      is_apple_m1_ = context.cl_context()->IsAppleM1();
      device_version =
          CLRuntime::Global()->device().getInfo<CL_DEVICE_VERSION>();
      y_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
      if (!is_mali_ && !is_apple_m1_ && x_dims.size() == 2 &&
          y_dims.size() == 2 && !transpose_x_) {
        build_options_ += " -DUSE_IMAGE_Y ";
        if (device_version.find("Adreno(TM) 506") == std::string::npos) {
          build_options_ += " -DADRENO_HIGH ";
        }
        use_image_y_ = true;
        DDimLite trans_dims{{y_ext_dims[0] / 4, y_ext_dims[1] * 4}};
        CLImageConverterFolder converter;
        const DDim& image_dims = converter.InitImageDimInfoWith(trans_dims);
        int image_w_ = image_dims[0];
        int image_h_ = image_dims[1];
        MUTABLE_DATA_GPU(y_gpu_t_, image_w_, image_h_, y_buffer_data);
      } else {
        auto y_gpu_data =
            y_gpu_t_->mutable_data(TARGET(kOpenCL), y_cpu_t->memory_size());
        TargetWrapperCL::MemcpySync(y_gpu_data,
                                    y_cpu_t->raw_data(),
                                    y_cpu_t->memory_size(),
                                    IoDirection::HtoD);
      }
    } else {
      // for y_persistable is false!!!
    }
    // reset to original fp16 precision
    if (precision_forced_to_fp32) {
      CLRuntime::Global()->set_precision(lite_api::CL_PRECISION_FP16);
    }
  }

  void ReInitWhenNeeded() override {
    const auto x_dims = matmul_v2_param_->X->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;
      // compute m,n,k
      const auto y_dims = matmul_v2_param_->Y->dims();
      const auto out_dims = matmul_v2_param_->Out->dims();
#ifdef LITE_WITH_LOG
      VLOG(4) << "x_dims:" << x_dims;
      VLOG(4) << "y_dims:" << y_dims;
      VLOG(4) << "out_dims:" << out_dims;
      VLOG(4) << "transpose_X:" << transpose_x_;
      VLOG(4) << "transpose_Y:" << transpose_y_;
#endif
      if (matmul_v2_param_->Y->persistable()) {
        if (x_dims.size() == 2 && y_dims.size() == 2) {
          m_ = transpose_x_ ? x_dims[1] : x_dims[0];
          k_ = transpose_x_ ? x_dims[0] : x_dims[1];
          n_ = transpose_y_ ? y_dims[0] : y_dims[1];
          kernel_func_name_ = "matmul";
          kernel_file_name_ = "image/matmul_opt_kernel.cl";
          if (transpose_x_) {
            kernel_func_name_ = "matmul_transpose_x";
            kernel_file_name_ = "image/matmul_xtranspose_kernel.cl";
          }
        } else if (x_dims.size() == 1 && y_dims.size() == 1 &&
                   x_dims[0] == y_dims[0]) {
          CHECK(transpose_x_ == transpose_y_)
              << "unsupported when x, y transpose is not equal";
          m_ = 1, n_ = 1;
          k_ = y_dims[0];
          kernel_func_name_ = "matmul";
          kernel_file_name_ = "image/matmul_opt_kernel.cl";
        } else if (x_dims.size() == 1 && y_dims.size() == 1 &&
                   x_dims[0] != y_dims[0]) {
          CHECK_EQ(transpose_x_, true)
              << "unsupported when x_transpose is false";
          CHECK_EQ(transpose_y_, true)
              << "unsupported when y_transpose is false";
          m_ = x_dims[0], n_ = y_dims[0];
          k_ = 1;
          kernel_func_name_ = "matmul_transpose_x";
          kernel_file_name_ = "image/matmul_xtranspose_kernel.cl";
        } else if (x_dims.size() >= 2 && y_dims.size() == 1 &&
                   x_dims[x_dims.size() - 1] == y_dims[0]) {
          m_ = 1, n_ = 1;
          k_ = y_dims[0];
          N = x_dims.size() == 4 ? x_dims[0] : 1;
          C = x_dims.size() == 4 ? x_dims[1]
                                 : (x_dims.size() == 3 ? x_dims[0] : 1);
          H = x_dims[x_dims.size() - 2], W = x_dims[x_dims.size() - 1];
          c_blks_ =
              x_dims.size() == 2 ? 1 : UP_DIV(x_dims[x_dims.size() - 3], 4);
          kernel_func_name_ = x_dims.size() == 4
                                  ? "matmul_xdim4_ydim1"
                                  : (x_dims.size() == 3 ? "matmul_xdim3_ydim1"
                                                        : "matmul_xdim2_ydim1");
          kernel_file_name_ = "image/matmul_kernel.cl";
        } else if (x_dims.size() > 2 && y_dims.size() == 2) {
          N = x_dims.size() == 4 ? x_dims[0] : 1;
          C = x_dims.size() == 4 ? x_dims[1] : x_dims[0];
          H = x_dims[x_dims.size() - 2], W = x_dims[x_dims.size() - 1];
          c_blks_ = UP_DIV(x_dims[x_dims.size() - 3], 4);
          batch_ = x_dims.count(0, x_dims.size() - 2);
          if ((!transpose_x_) && (!transpose_y_)) {
            m_ = x_dims[x_dims.size() - 2];
            n_ = y_dims[y_dims.size() - 1];
            k_ = x_dims[x_dims.size() - 1];
            kernel_func_name_ = "matmul_highdimx_ydim2";
            kernel_file_name_ = "image/matmul_kernel.cl";
          } else if ((!transpose_x_) && transpose_y_) {
            m_ = x_dims[x_dims.size() - 2];
            n_ = y_dims[y_dims.size() - 2];
            k_ = x_dims[x_dims.size() - 1];
            kernel_func_name_ = "matmul_highdimx_ydim2";
            kernel_file_name_ = "image/matmul_kernel.cl";
          } else if (transpose_x_ && (!transpose_y_)) {
            m_ = x_dims[x_dims.size() - 1];
            n_ = y_dims[y_dims.size() - 1];
            k_ = x_dims[x_dims.size() - 2];
            kernel_func_name_ = "matmul_highdimxtranspose_ydim2";
            kernel_file_name_ = "image/matmul_xtranspose_kernel.cl";
          } else if (transpose_x_ && transpose_y_) {
            m_ = x_dims[x_dims.size() - 1];
            n_ = y_dims[y_dims.size() - 2];
            k_ = x_dims[x_dims.size() - 2];
            kernel_func_name_ = "matmul_highdimxtranspose_ydim2";
            kernel_file_name_ = "image/matmul_xtranspose_kernel.cl";
          }
        } else if (x_dims.size() > 2 && y_dims.size() > 2) {
          N = x_dims.size() == 4 ? x_dims[0] : 1;
          c_blks_ = UP_DIV(x_dims[x_dims.size() - 3], 4);
          if ((!transpose_x_) && (!transpose_y_)) {
            m_ = x_dims[x_dims.size() - 2];
            n_ = y_dims[y_dims.size() - 1];
            k_ = x_dims[x_dims.size() - 1];
            kernel_func_name_ = "matmul_highdim";
            kernel_file_name_ = "image/matmul_kernel.cl";
          } else if ((!transpose_x_) && transpose_y_) {
            m_ = x_dims[x_dims.size() - 2];
            n_ = y_dims[y_dims.size() - 2];
            k_ = x_dims[x_dims.size() - 1];
            kernel_func_name_ = "matmul_highdim";
            kernel_file_name_ = "image/matmul_kernel.cl";
          } else if (transpose_x_ && (!transpose_y_)) {
            m_ = x_dims[x_dims.size() - 1];
            n_ = y_dims[y_dims.size() - 1];
            k_ = x_dims[x_dims.size() - 2];
            kernel_func_name_ = "matmul_highdim_transpose_x";
            kernel_file_name_ = "image/matmul_xtranspose_kernel.cl";
          } else {
            m_ = x_dims[x_dims.size() - 1];
            n_ = y_dims[y_dims.size() - 2];
            k_ = x_dims[x_dims.size() - 2];
            kernel_func_name_ = "matmul_highdim_transpose_x";
            kernel_file_name_ = "image/matmul_xtranspose_kernel.cl";
          }
        } else {
          LOG(FATAL) << "unsupported input case.";
        }

        k_blks_ = UP_DIV(k_, 4);
        n_blks_ = UP_DIV(n_, 4);
#ifdef LITE_WITH_LOG
        VLOG(4) << "batch:" << batch_ << ", m_:" << m_ << ", k_:" << k_
                << ", n_:" << n_;
#endif
      } else {
        // for y_persistable is false!!!
        if (x_dims.size() > 2 && y_dims.size() > 2) {
          N = x_dims.size() == 4 ? x_dims[0] : 1;
          c_blks_ = UP_DIV(x_dims[x_dims.size() - 3], 4);
          if ((!transpose_x_) && (!transpose_y_)) {
            m_ = x_dims[x_dims.size() - 2];
            n_ = y_dims[y_dims.size() - 1];
            k_ = x_dims[x_dims.size() - 1];
            kernel_func_name_ = "matmul";
            kernel_file_name_ = "image/matmul_unpersistable_y_kernel.cl";
          } else if ((!transpose_x_) && transpose_y_) {
            m_ = x_dims[x_dims.size() - 2];
            n_ = y_dims[y_dims.size() - 2];
            k_ = x_dims[x_dims.size() - 1];
            kernel_func_name_ = "matmul_ytranspose";
            kernel_file_name_ = "image/matmul_unpersistable_y_kernel.cl";
          } else if (transpose_x_ && (!transpose_y_)) {
            m_ = x_dims[x_dims.size() - 1];
            n_ = y_dims[y_dims.size() - 1];
            k_ = x_dims[x_dims.size() - 2];
            kernel_func_name_ = "matmul_xtranspose";
            kernel_file_name_ = "image/matmul_unpersistable_y_kernel.cl";
          } else {
            m_ = x_dims[x_dims.size() - 1];
            n_ = y_dims[y_dims.size() - 2];
            k_ = x_dims[x_dims.size() - 2];
            kernel_func_name_ = "matmul_xytranspose";
            kernel_file_name_ = "image/matmul_unpersistable_y_kernel.cl";
          }
        } else {
          LOG(FATAL) << "unsupported input case.";
        }
      }
    }
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(
        kernel_func_name_, kernel_file_name_, build_options_, time_stamp_);
    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());

    SetGlobalLocalWorkSize();
  }

  void SetGlobalLocalWorkSize() {
    const auto x_dims = matmul_v2_param_->X->dims();
    const auto y_dims = matmul_v2_param_->Y->dims();
    // compute global/local work size
    auto device_info = CLRuntime::Global()->GetDeviceInfo();
    int max_work_group_size = device_info["CL_DEVICE_MAX_WORK_GROUP_SIZE"];
    CLImageConverterFolder* folder_converter = new CLImageConverterFolder();
    const auto out_dims = matmul_v2_param_->Out->dims();
    out_img_shape = folder_converter->InitImageDimInfoWith(out_dims);

    if (matmul_v2_param_->Y->persistable()) {
      if (x_dims.size() == 2 && y_dims.size() == 1) {
        local_work_size_ = cl::NDRange(1, 1);
        global_work_size_ = cl::NDRange(UP_DIV(H, 4), c_blks_);
      } else if (x_dims.size() <= 2 && y_dims.size() <= 2) {
        if (transpose_x_) {
          local_work_size_ = cl::NDRange(32, 4, 1);
          global_work_size_ =
              cl::NDRange(ROUND_UP(UP_DIV(n_, 4), local_work_size_[0]),
                          local_work_size_[1],
                          UP_DIV(m_, 4));
        } else {
          local_work_size_ = cl::NDRange(8, 4, 16);
          if (device_version.find("Adreno(TM) 506") != std::string::npos) {
            local_work_size_ = cl::NDRange(4, 4, 16);
          }
          global_work_size_ =
              cl::NDRange(m_, local_work_size_[1], UP_DIV(n_, 4));
          if (is_mali_ || is_apple_m1_) {
            local_work_size_ = cl::NDRange(4, 4, 16);
            global_work_size_ =
                cl::NDRange(ROUND_UP(m_, local_work_size_[0]),
                            local_work_size_[1],
                            ROUND_UP(UP_DIV(n_, 4), local_work_size_[2]));
          }
        }
      } else if (x_dims.size() > 2 && y_dims.size() >= 2) {
        local_work_size_ =
            cl::NDRange(32, std::min(c_blks_, max_work_group_size / 32), 1);
        global_work_size_ = cl::NDRange(ROUND_UP(n_, local_work_size_[0]),
                                        ROUND_UP(c_blks_, local_work_size_[1]),
                                        out_img_shape[1]);
      } else if (x_dims.size() > 2 && y_dims.size() == 1) {
        local_work_size_ =
            (x_dims.size() == 4)
                ? cl::NDRange(
                      32, std::min(c_blks_, max_work_group_size / 32), 1)
                : cl::NDRange(1, 1);
        global_work_size_ =
            (x_dims.size() == 4)
                ? cl::NDRange(ROUND_UP(H, local_work_size_[0]),
                              ROUND_UP(c_blks_, local_work_size_[1]),
                              UP_DIV(N, 4))
                : cl::NDRange(UP_DIV(H, 4), c_blks_);
      }
    } else {
      // for y_persistable is false!!!
      local_work_size_ = cl::NullRange;
      auto default_work_size =
          DefaultGlobalWorkSize(out_dims,
                                DDim(std::vector<DDim::value_type>{
                                    static_cast<int64_t>(out_img_shape[0]),
                                    static_cast<int64_t>(out_img_shape[1])}));
      global_work_size_ =
          cl::NDRange{static_cast<cl::size_type>(default_work_size[0]),
                      static_cast<cl::size_type>(default_work_size[1]),
                      static_cast<cl::size_type>(default_work_size[2])};
    }
    VLOG(4) << "local_work_size[3D]: " << local_work_size_[0] << " "
            << local_work_size_[1] << " " << local_work_size_[2];
    VLOG(4) << "global_work_size[3D]: " << global_work_size_[0] << " "
            << global_work_size_[1] << " " << global_work_size_[2];
  }

  void Run() override {
    auto* x_img_ = GET_DATA_GPU(matmul_v2_param_->X);
    auto* out_img_ = MUTABLE_DATA_GPU(
        matmul_v2_param_->Out, out_img_shape[0], out_img_shape[1], nullptr);

    auto x_dims = matmul_v2_param_->X->dims();
    auto y_dims = matmul_v2_param_->Y->dims();

    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);

    cl_int status;
    int arg_idx = 0;
    auto kernel = kernel_;
    status = kernel.setArg(arg_idx++, *x_img_);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *out_img_);
    CL_CHECK_FATAL(status);
    if (matmul_v2_param_->Y->persistable()) {
      if (!use_image_y_) {
        auto* y_buf_ = GET_BUFFER_GPU(y_gpu_t_);
        status = kernel.setArg(arg_idx++, *y_buf_);
        CL_CHECK_FATAL(status);
      } else {
        auto* y_img_ = GET_DATA_GPU(y_gpu_t_);
        status = kernel.setArg(arg_idx++, *y_img_);
        CL_CHECK_FATAL(status);
      }
      status = kernel.setArg(arg_idx++, m_);
      CL_CHECK_FATAL(status);
      if (x_dims.size() == 2 && y_dims.size() == 1) {
        status = kernel.setArg(arg_idx++, C);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(arg_idx++, H);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(arg_idx++, W);
        CL_CHECK_FATAL(status);
      } else if (x_dims.size() <= 2 && y_dims.size() <= 2) {
        status = kernel.setArg(arg_idx++, k_blks_);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(arg_idx++, n_blks_);
        CL_CHECK_FATAL(status);
      } else if (x_dims.size() > 2 && y_dims.size() >= 2) {
        status = kernel.setArg(arg_idx++, k_);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(arg_idx++, n_);
        CL_CHECK_FATAL(status);
        int out_image_width = out_img_shape[0];
        status = kernel.setArg(arg_idx++, out_image_width);
        CL_CHECK_FATAL(status);
      } else if (x_dims.size() > 2 && y_dims.size() == 1) {
        status = kernel.setArg(arg_idx++, C);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(arg_idx++, H);
        CL_CHECK_FATAL(status);
        status = kernel.setArg(arg_idx++, W);
        CL_CHECK_FATAL(status);
      }
    } else {
      // for y_persistable is false!!!
      auto* y_img_ = GET_DATA_GPU(matmul_v2_param_->Y);
      auto out_dims = matmul_v2_param_->Out->dims();
      int out_width = out_dims[out_dims.size() - 1];
      int out_height = out_dims[out_dims.size() - 2];
      status = kernel.setArg(arg_idx++, *y_img_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, k_);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, out_width);
      CL_CHECK_FATAL(status);
      status = kernel.setArg(arg_idx++, out_height);
      CL_CHECK_FATAL(status);
    }
    status = kernel.setArg(arg_idx++, alpha_);

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size_,
                                  local_work_size_,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }

  void convert(const float* nchw, void* image, const DDim& tensor_dim) {
    size_t new_dims[] = {1, 1, 1, 1};
    for (size_t j = 0; j < tensor_dim.size(); ++j) {
      new_dims[4 - tensor_dim.size() + j] = tensor_dim[j];
    }

    size_t N, C, H, W;
    N = new_dims[0];
    C = new_dims[1];
    H = new_dims[2];
    W = new_dims[3];

    size_t width = W * UP_DIV(C, 4);
    size_t w_block = width / W;

    float* image_fp32 = static_cast<float*>(image);
    half_t* image_fp16 = static_cast<half_t*>(image);

    size_t i0 = 0;
    size_t index = 0;
    for (size_t n = 0; n < N; n++) {
      for (size_t c = 0; c < w_block * 4; c++) {
        size_t i1 = i0 + (c / 4) * W;
        for (size_t h = 0; h < H; h++) {
          size_t i2 = (i1 << 2) + c % 4;
          for (size_t w = 0; w < W; w++) {
            if (c < C) {
              fp16_support_ ? image_fp16[i2] = Float2Half(nchw[index++])
                            : image_fp32[i2] = nchw[index++];
              i2 += 4;
            } else {
              fp16_support_ ? image_fp16[i2] = Float2Half(0.f)
                            : image_fp32[i2] = 0.f;
              i2 += 4;
            }
          }
          i1 += width;
        }
      }
      i0 += width * H;
    }
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->global_work_size = ch->NDRangeToStr(global_work_size_);
    ch->local_work_size = ch->NDRangeToStr(local_work_size_);
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  int m_{0};
  int n_{0};
  int k_{0};
  int batch_{1};
  int lws_{1};
  int N{1};
  int C{1};
  int H{1};
  int W{1};
  int c_blks_;
  int k_blks_;
  int n_blks_;
  bool is_mali_;
  bool is_apple_m1_;
  std::string device_version;
  bool use_image_y_{false};
  bool transpose_x_{false};
  bool transpose_y_{false};
  float alpha_{1.0f};
  param_t* matmul_v2_param_{nullptr};
  std::string kernel_func_name_{};
  std::string kernel_file_name_{};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  DDim out_img_shape;
  std::unique_ptr<Tensor> y_gpu_t_{nullptr};

  cl::NDRange global_work_size_;
  cl::NDRange local_work_size_;
  cl::Kernel kernel_;
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(matmul,
                     kOpenCL,
                     kFP16,
                     kImageFolder,
                     paddle::lite::kernels::opencl::MatMulV2ImageCompute,
                     image2d_host)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageFolder))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageFolder))})
    .Finalize();

REGISTER_LITE_KERNEL(matmul,
                     kOpenCL,
                     kFP16,
                     kImageFolder,
                     paddle::lite::kernels::opencl::MatMulV2ImageCompute,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageFolder))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageFolder))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageFolder))})
    .Finalize();

REGISTER_LITE_KERNEL(matmul_v2,
                     kOpenCL,
                     kFP16,
                     kImageFolder,
                     paddle::lite::kernels::opencl::MatMulV2ImageCompute,
                     image2d)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageFolder))})
    .BindInput("Y",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageFolder))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageFolder))})
    .Finalize();

REGISTER_LITE_KERNEL(matmul_v2,
                     kOpenCL,
                     kFP16,
                     kImageFolder,
                     paddle::lite::kernels::opencl::MatMulV2ImageCompute,
                     image2d_host)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageFolder))})
    .BindInput("Y", {LiteType::GetTensorTy(TARGET(kHost))})
    .BindOutput("Out",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageFolder))})
    .Finalize();
