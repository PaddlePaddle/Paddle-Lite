// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //     http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.

#include <gtest/gtest.h>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/image_helper.h"

namespace paddle {
namespace lite {

template <typename dtype>
void concat2_compute_ref(const dtype *in0,
                         const dtype *in1,
                         const int axis,
                         const DDim in0_dim,
                         const DDim in1_dim,
                         const DDim out_dim,
                         dtype *out_data) {
  int pre_size = 1;
  int post_size = 1;
  for (int i = 0; i < axis; i++) {
    pre_size *= in0_dim[i];
  }
  for (int i = axis + 1; i < in0_dim.size(); i++) {
    post_size *= in0_dim[i];
  }
  int axis_size = out_dim[axis];
  for (int i = 0; i < pre_size; i++) {
    for (int j = 0; j < axis_size; j++) {
      if (j < in0_dim[axis]) {
        memcpy(out_data, in0, sizeof(dtype) * post_size);
        in0 += post_size;
        out_data += post_size;
      }
    }
  }
}

template <typename dtype>
void concat_mul_compute_ref(std::vector<const dtype *> ins_data,
                            std::vector<const DDim> ins_dim,
                            int axis,
                            const DDim out_dim,
                            dtype *out_data) {
  int pre_size = 1;
  int post_size = 1;
  for (int i = 0; i < axis; i++) {
    pre_size *= ins_dim[0][i];
  }
  for (int i = axis + 1; i < ins_dim[0].size(); i++) {
    post_size *= ins_dim[0][i];
  }
  int axis_size = out_dim[axis];
  for (int i = 0; i < pre_size; i++) {
    for (int j = 0; j < ins_data.size(); j++) {
      int size = post_size * ins_dim[j][axis];
      memcpy(out_data, ins_data[j], sizeof(dtype) * size);
      out_data += size;
    }
  }
}

TEST(opencl_concat_buffer, compute) {
  // prepare data
  const DDim x0_dim = DDim(std::vector<DDim::value_type>{1, 2, 3, 4});
  const DDim x1_dim = DDim(std::vector<DDim::value_type>{1, 2, 3, 4});
  const DDim x2_dim = DDim(std::vector<DDim::value_type>{1, 2, 3, 4});
  const DDim out_dim = DDim(std::vector<DDim::value_type>{1, 6, 3, 4});
  lite::Tensor x0, x1, x2, out, out_ref;
  x0.Resize(x0_dim);
  x1.Resize(x1_dim);
  x2.Resize(x2_dim);
  out.Resize(out_dim);
  out_ref.Resize(out_dim);

  auto *x0_data = x0.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto *x1_data = x1.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  auto *x2_data = x2.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
  std::default_random_engine engine;
  std::uniform_real_distribution<float> dist(-10, 10);
  auto *mapped_x0 = static_cast<float *>(
      TargetWrapperCL::Map(x0_data, 0, sizeof(float) * x0_dim.production()));
  auto *mapped_x1 = static_cast<float *>(
      TargetWrapperCL::Map(x1_data, 0, sizeof(float) * x1_dim.production()));
  auto *mapped_x2 = static_cast<float *>(
      TargetWrapperCL::Map(x2_data, 0, sizeof(float) * x2_dim.production()));
  for (int i = 0; i < x0_dim.production(); i++) {
    mapped_x0[i] = i + 1;  // dist(engine);
  }
  for (int i = 0; i < x1_dim.production(); i++) {
    mapped_x1[i] = x0_dim.production() + i + 1;  // dist(engine);
  }
  for (int i = 0; i < x2_dim.production(); i++) {
    mapped_x2[i] =
        x0_dim.production() + x1_dim.production() + i + 1;  // dist(engine);
  }

  // set param and kernel, then run
  operators::ConcatParam param;
  std::vector<lite::Tensor *> ins;
  ins.push_back(&x0);
  ins.push_back(&x1);
  ins.push_back(&x2);
  auto axis = 1;
  param.x = ins;
  param.output = &out;
  param.axis = axis;

  std::vector<const float *> ins_data;
  std::vector<const DDim> ins_dim;

  ins_data.push_back(mapped_x0);
  ins_data.push_back(mapped_x1);
  ins_data.push_back(mapped_x2);
  ins_dim.push_back(x0_dim);
  ins_dim.push_back(x1_dim);
  ins_dim.push_back(x2_dim);

  std::unique_ptr<KernelContext> context(new KernelContext);
  context->As<OpenCLContext>().InitOnce();
  auto kernels = KernelRegistry::Global().Create(
      "concat", TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW));
  ASSERT_FALSE(kernels.empty());
  auto kernel = std::move(kernels.front());
  kernel->SetParam(param);
  std::unique_ptr<KernelContext> concat_context(new KernelContext);
  context->As<OpenCLContext>().CopySharedTo(
      &(concat_context->As<OpenCLContext>()));
  kernel->SetContext(std::move(concat_context));
  kernel->Launch();

  CLRuntime::Global()->command_queue().finish();

  // run compute ref and check
  auto *out_ref_data = out_ref.mutable_data<float>(TARGET(kARM));
  concat_mul_compute_ref<float>(ins_data, ins_dim, axis, out_dim, out_ref_data);

  auto *out_data = out.mutable_data<float, cl::Buffer>();
  auto *mapped_out = static_cast<float *>(
      TargetWrapperCL::Map(out_data, 0, sizeof(float) * out_dim.production()));
#ifdef PRINT_RESULT_CONCAT_BUFFER
  for (int i = 0; i < out_dim.production(); i++) {
    LOG(INFO) << "i:" << i << ", out[" << i << "]:" << mapped_out[i]
              << ", out_ref_data[" << i << "]:" << out_ref_data[i];
  }
#endif
  EXPECT_NEAR(mapped_out[i], out_ref_data[i], 1e-6);
  TargetWrapperCL::Unmap(out_data, mapped_out);
  TargetWrapperCL::Unmap(x0_data, mapped_x0);
  TargetWrapperCL::Unmap(x1_data, mapped_x1);
  TargetWrapperCL::Unmap(x2_data, mapped_x2);
}

}  // namespace lite
}  // namespace paddle

// concat buffer
USE_LITE_KERNEL(concat, kOpenCL, kFloat, kNCHW, def);
