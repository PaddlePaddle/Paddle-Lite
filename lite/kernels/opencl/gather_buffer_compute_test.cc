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
#include <cmath>
#include <ctime>
#include <iostream>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"

namespace paddle {
namespace lite {

template <typename Dtype>
void fill_input_rand(Dtype *data, size_t size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(0, 1.f);
  for (size_t i = 0; i < size; ++i) {
    data[i] = i;
  }
}

void fill_axis_rand(int32_t *axis, size_t length, int x_size) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 10);
  for (int i = 0; i < length; ++i) {
    axis[i] = dis(gen) % x_size;  // 0 ~ x_size;
  }
}

template <typename Dtype>
void fill_index_rand(Dtype *index, size_t length, int mid_dim) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<int> dis(0, 100);
  for (int i = 0; i < length; ++i) {
    index[i] = dis(gen) % mid_dim;  // 0 ~ mid_dim;
  }
}

template <typename Dtype>
void gather_compute_without_axis(
    Dtype *in, Dtype *out, Dtype *index, int index_size, int slice_size) {
  for (int i = 0; i < index_size; ++i) {
    int index_ = (int)index[i];
    memcpy(out + i * slice_size,
           in + index_ * slice_size,
           slice_size * sizeof(Dtype));
  }
}

template <typename Dtype>
void gather_compute_with_axis(Dtype *in,
                              Dtype *out,
                              Dtype *index,
                              int index_size,
                              int input_size,
                              int inner_dim_size,
                              int outer_dim_size) {
  int out_index = 0;
  for (int i = 0; i < inner_dim_size; i++) {
    for (int j = 0; j < index_size; j++) {
      for (int k = 0; k < outer_dim_size; k++) {
        int index_ = (int)index[j];
        int in_index =
            k + index_ * outer_dim_size + (i * input_size / inner_dim_size);
        out[out_index] = in[in_index];
        out_index++;
      }
    }
  }
}

TEST(gather_buffer, compute) {
  std::vector<DDim> x_dim_v{DDim(std::vector<DDim::value_type>{5, 7, 3, 4}),
                            DDim(std::vector<DDim::value_type>{1, 2, 4096}),
                            DDim(std::vector<DDim::value_type>{12, 17})};
  std::vector<DDim> axis_dim_v{DDim(std::vector<DDim::value_type>{1}),
                               DDim(std::vector<DDim::value_type>{0})};
  std::vector<DDim> index_dim_v{DDim(std::vector<DDim::value_type>{3}),
                                DDim(std::vector<DDim::value_type>{10}),
                                DDim(std::vector<DDim::value_type>{100})};

  for (int i = 0; i < axis_dim_v; ++i) {
    for (int j = 0; j < x_dim_v; ++j) {
      for (int k = 0; k < index_dim_v; ++k) {
        auto x_dim = x_dim_v[j];
        auto axis_dim = axis_dim_v[i];
        auto index_dim = index_dim_v[k];

        lite::Tensor gather_x, gather_index, gather_axis, gather_out;
        gather_x.Resize(x_dim);
        gather_index.Resize(index_dim);

        auto *x_data =
            gather_x.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
        auto *mapped_x = static_cast<float *>(TargetWrapperCL::Map(
            x_data, 0, sizeof(float) * x_dim.production()));
        fill_input_rand(mapped_x, x_dim.production());
        auto *index_data =
            gather_index.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
        auto *mapped_index = static_cast<float *>(TargetWrapperCL::Map(
            index_data, 0, sizeof(float) * index_dim.production()));

        if (axis_dim.production() != 0) {
          VLOG(4) << "Axis is not null!";
          gather_axis.Resize(axis_dim);
          auto *axis_data = gather_axis.mutable_data<int32_t>();
          fill_axis_rand(axis_data, axis_dim.production(), x_dim.size());
          int axis_index = axis_data[0];
          fill_index_rand(
              mapped_index, index_dim.production(), x_dim[axis_index]);
          int inner_dim_size = 1;
          int outer_dim_size = 1;
          int index_size = index_dim.production();
          VLOG(4) << "axis_index = " << axis_index;
          VLOG(4) << "x_dim[axis_index] = " << x_dim[axis_index];
          VLOG(4) << "index_size = " << index_size;
          VLOG(4) << "index_data: ";
          for (int i = 0; i < index_size; ++i) {
            std::cout << mapped_index[i] << " ";
          }
          int input_size = x_dim.production();
          auto out_dim = x_dim;
          out_dim[axis_index] = index_size;
          for (int i = 0; i < axis_index; i++) {
            inner_dim_size *= x_dim[i];
          }
          for (int i = axis_index + 1; i < x_dim.size(); i++) {
            outer_dim_size *= x_dim[i];
          }
          gather_out.Resize(out_dim);
          auto *out_data =
              gather_out.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
          auto *mapped_out = static_cast<float *>(TargetWrapperCL::Map(
              out_data, 0, sizeof(float) * out_dim.production()));

          auto gather_buf_kernels = KernelRegistry::Global().Create(
              "gather", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
          ASSERT_FALSE(gather_buf_kernels.empty());
          auto kernel = std::move(gather_buf_kernels.front());

          VLOG(4) << "set context and kernel args";
          operators::GatherParam gatherParam;
          gatherParam.X = &gather_x;
          gatherParam.Index = &gather_index;
          gatherParam.Axis = &gather_axis;
          gatherParam.Out = &gather_out;
          std::unique_ptr<KernelContext> context(new KernelContext);
          context->As<OpenCLContext>().InitOnce();

          kernel->SetParam(gatherParam);
          std::unique_ptr<KernelContext> gather_buf_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(gather_buf_context->As<OpenCLContext>()));
          kernel->SetContext(std::move(gather_buf_context));

          VLOG(4) << "run kernel";
          kernel->Launch();

          CLRuntime::Global()->command_queue().finish();
          std::cout << "finish executing kernel..." << std::endl;
          // compute cpu reference
          std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
          gather_compute_with_axis(mapped_x,
                                   out_ref.get(),
                                   mapped_index,
                                   index_size,
                                   input_size,
                                   inner_dim_size,
                                   outer_dim_size);

          // compare cpu gpu results
          for (int eidx = 0; eidx < out_dim.production(); eidx++) {
            auto value = mapped_out[eidx];
            auto ref_value = out_ref.get()[eidx];
            auto diff = abs(value - ref_value);
            if (diff != 0.f) {
              std::cout << "diff in this case at eidx[from 0]:" << eidx << " / "
                        << out_dim.production() << ", value[" << eidx
                        << "]:" << value << ", ref_value[" << eidx
                        << "]:" << ref_value << std::endl;
            }
          }
          TargetWrapperCL::Unmap(out_data, mapped_out);
        } else {
          VLOG(4) << "Axis is null!";
          fill_index_rand(mapped_index, index_dim.production(), x_dim[0]);
          int index_size = index_dim.production();
          VLOG(4) << "index_size = " << index_size;
          VLOG(4) << "index_data: ";
          for (int i = 0; i < index_size; ++i) {
            std::cout << mapped_index[i] << " ";
          }
          auto out_dim = x_dim;
          int batch_size = index_dim[0];
          out_dim[0] = batch_size;
          gather_out.Resize(out_dim);
          auto *out_data =
              gather_out.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
          auto *mapped_out = static_cast<float *>(TargetWrapperCL::Map(
              out_data, 0, sizeof(float) * out_dim.production()));

          auto gather_buf_kernels = KernelRegistry::Global().Create(
              "gather", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
          ASSERT_FALSE(gather_buf_kernels.empty());
          auto kernel = std::move(gather_buf_kernels.front());

          VLOG(4) << "set context and kernel args";
          operators::GatherParam gatherParam;
          gatherParam.X = &gather_x;
          gatherParam.Index = &gather_index;
          gatherParam.Axis = {nullptr};
          gatherParam.Out = &gather_out;
          std::unique_ptr<KernelContext> context(new KernelContext);
          context->As<OpenCLContext>().InitOnce();

          kernel->SetParam(gatherParam);
          std::unique_ptr<KernelContext> gather_buf_context(new KernelContext);
          context->As<OpenCLContext>().CopySharedTo(
              &(gather_buf_context->As<OpenCLContext>()));
          kernel->SetContext(std::move(gather_buf_context));

          VLOG(4) << "run kernel";
          kernel->Launch();

          CLRuntime::Global()->command_queue().finish();

          // compute cpu reference
          std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);
          int slice_size = 1;
          for (size_t i = 1; i < x_dim.size(); ++i) {
            slice_size *= x_dim[i];
          }
          gather_compute_without_axis(
              mapped_x, out_ref.get(), mapped_index, batch_size, slice_size);

          // compare cpu gpu results
          for (int eidx = 0; eidx < out_dim.production(); eidx++) {
            auto value = mapped_out[eidx];
            auto ref_value = out_ref.get()[eidx];
            auto diff = abs(value - ref_value);
            if (diff != 0.f) {
              std::cout << "diff in this case at eidx[from 0]:" << eidx << " / "
                        << out_dim.production() << ", value[" << eidx
                        << "]:" << value << ", ref_value[" << eidx
                        << "]:" << ref_value << std::endl;
            }
          }
          TargetWrapperCL::Unmap(out_data, mapped_out);
        }
        TargetWrapperCL::Unmap(index_data, mapped_index);
        TargetWrapperCL::Unmap(x_data, mapped_x);
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(gather, kOpenCL, kFloat, kNCHW, def);
