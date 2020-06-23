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
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"

#define FP16_MAX_DIFF (5e-1)

namespace paddle {
namespace lite {

template <typename dtype>
void softmax_compute_ref(const operators::SoftmaxParam& param) {
  const dtype* x_data = param.x->mutable_data<const dtype>();
  dtype* output_data = param.output->mutable_data<dtype>();
  DDim x_dims = param.x->dims();
  ASSERT_EQ(x_dims.data(), param.output->dims().data());
  auto x_rank = x_dims.size();
  int axis = param.axis;
  if (axis < 0) {
    axis += x_rank;
  }
  int axis_size = x_dims[axis];
  int outer_num = x_dims.Slice(0, axis).production();
  int inner_num = x_dims.Slice(axis + 1, x_rank).production();
  int compute_size = outer_num * inner_num;
  for (int i = 0; i < compute_size; i++) {
    int idx_inner = i % inner_num;
    int idx_outer = (i / inner_num) * axis_size;
    int start = idx_outer * inner_num + idx_inner;
    int offset;

    offset = start;
    dtype max_data = std::numeric_limits<dtype>::lowest();
    for (int j = 0; j < axis_size; j++) {
      max_data = x_data[offset] > max_data ? x_data[offset] : max_data;
      offset += inner_num;
    }

    offset = start;
    dtype sum_data = (dtype)0;
    for (int j = 0; j < axis_size; j++) {
      output_data[offset] = exp(x_data[offset] - max_data);
      sum_data += output_data[offset];
      offset += inner_num;
    }

    offset = start;
    for (int j = 0; j < axis_size; j++) {
      output_data[offset] /= sum_data;
      offset += inner_num;
    }
  }
}

TEST(softmax_image2d, compute) {
#if 1
  for (auto n : {1, 3}) {
    for (auto c : {1, 4}) {
      for (auto h : {5, 1}) {
        for (auto w : {1, 6}) {
          for (auto axis : {/*-2,*/ -1 /*, 0, 1, 2*/}) {
#else
  for (auto n : {1, 3, 4, 11}) {
    for (auto c : {1, 3, 11, 4}) {
      for (auto h : {3, 1, 11, 4}) {
        for (auto w : {1, 3, 4, 12}) {
          for (auto axis : {-4, -3, -2, -1, 0, 1, 2, 3}) {
#endif
            LOG(INFO) << "create kernel ...";
            auto kernels =
                KernelRegistry::Global().Create("softmax",
                                                TARGET(kOpenCL),
                                                PRECISION(kFP16),
                                                DATALAYOUT(kImageDefault));
            ASSERT_FALSE(kernels.empty());
            // prepare opencl kernel params
            auto kernel = std::move(kernels.front());
            LOG(INFO) << "prepare to test kernel ====> " << kernel->doc();
            LOG(INFO) << n << c << h << w;
            operators::SoftmaxParam param;
            lite::Tensor x;
            lite::Tensor output;

            operators::SoftmaxParam param_ref;
            lite::Tensor x_ref;
            lite::Tensor output_ref;

            auto in_dim = DDim(std::vector<int64_t>({n, c, h, w}));
            auto out_dim = DDim(std::vector<int64_t>({n, c, h, w}));
            x.Resize(in_dim);
            x_ref.Resize(in_dim);

            output.Resize(out_dim);
            output_ref.Resize(out_dim);

            param.x = &x;
            param.axis = axis;
            param.output = &output;

            param_ref.x = &x_ref;
            param_ref.axis = axis;
            param_ref.output = &output_ref;
            auto* x_ref_data = x_ref.mutable_data<float>();

            std::unique_ptr<KernelContext> context(new KernelContext);
            context->As<OpenCLContext>().InitOnce();

            kernel->SetParam(param);
            std::unique_ptr<KernelContext> softmax_context(new KernelContext);
            context->As<OpenCLContext>().CopySharedTo(
                &(softmax_context->As<OpenCLContext>()));

            kernel->SetContext(std::move(softmax_context));

            std::default_random_engine engine;
            std::uniform_real_distribution<float> dist(-2, 2);
            std::vector<float> input_v(n * c * h * w);

            int index = 0;
            for (auto& i : input_v) {
              x_ref_data[index] = index;
              i = index++;
            }
            VLOG(1) << "input_v ..... ";
            for (size_t i = 0; i < input_v.size(); i++) {
              VLOG(10) << input_v[i];
            }

            LOG(INFO) << "prepare input";
            CLImageConverterDefault* default_converter =
                new CLImageConverterDefault();
            DDim x_image_shape = default_converter->InitImageDimInfoWith(
                DDim(std::vector<int64_t>({n, c, h, w})));
            LOG(INFO) << "x_image_shape = " << x_image_shape[0] << " "
                      << x_image_shape[1];
            std::vector<half_t> x_image_data(x_image_shape.production() *
                                             4);  // 4 : RGBA
            default_converter->NCHWToImage(
                input_v.data(), x_image_data.data(), in_dim);
            auto* x_image = x.mutable_data<half_t, cl::Image2D>(
                x_image_shape[0], x_image_shape[1], x_image_data.data());
            VLOG(1) << "x_image_data ..... ";
            for (size_t i = 0; i < x_image_data.size(); i++) {
              VLOG(10) << Half2Float(x_image_data[i]);
            }
            DDim out_image_shape =
                default_converter->InitImageDimInfoWith(out_dim);
            LOG(INFO) << "out_image_shape = " << out_image_shape[0] << " "
                      << out_image_shape[1];
            auto* out_image = output.mutable_data<half_t, cl::Image2D>(
                out_image_shape[0], out_image_shape[1]);
            // run
            kernel->Launch();
            CLRuntime::Global()->command_queue().finish();

            // handle output
            const size_t cl_image2d_row_pitch{0};
            const size_t cl_image2d_slice_pitch{0};

            std::vector<half_t> out_image_v(out_image_shape.production() * 4);
            half_t* out_image_data = out_image_v.data();
            TargetWrapperCL::ImgcpySync(out_image_data,
                                        out_image,
                                        out_image_shape[0],
                                        out_image_shape[1],
                                        cl_image2d_row_pitch,
                                        cl_image2d_slice_pitch,
                                        IoDirection::DtoH);
            VLOG(1) << "out_image_data ..... ";
            for (size_t i = 0; i < out_image_shape.production() * 4; i++) {
              VLOG(10) << Half2Float(out_image_data[i]);
            }
            std::vector<float> out_data(out_image_shape.production() * 4);
            default_converter->ImageToNCHW(
                out_image_data, out_data.data(), out_image_shape, out_dim);

            VLOG(1) << "out_data ..... ";
            for (int i = 0; i < out_dim.production(); i++) {
              VLOG(10) << out_data[i];
            }

            auto* output_ref_data = output_ref.mutable_data<float>();
            softmax_compute_ref<float>(param_ref);

            for (int i = 0; i < output.dims().production(); i++) {
              EXPECT_NEAR(out_data[i], output_ref_data[i], 1e-2);
            }
          }
        }
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(softmax, kOpenCL, kFP16, kImageDefault, image2d);
