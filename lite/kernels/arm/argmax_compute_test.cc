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

#include <cstdlib>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "lite/core/op_registry.h"
#include "lite/kernels/arm/argmax_compute.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

template <typename dtype>
void argmax_compute_ref(const operators::ArgmaxParam& param) {
  lite::Tensor* x = param.X;
  lite::Tensor* output = param.Out;
  int axis = param.Axis;

  auto x_data = x->data<dtype>();
  auto output_data = output->mutable_data<int64_t>();
  DDim x_dims = x->dims();
  DDim output_dims = output->dims();

  // int in_channel = x_dims
  const int size = x_dims[axis];
  const int in_channel = x_dims.count(axis, x_dims.size());
  const int out_channel = output_dims.count(axis, output_dims.size());
  const int in_stride = x_dims.count(axis + 1, x_dims.size());
  const int out_stride = x_dims.count(0, axis);

  for (int n = 0; n < out_stride; n++) {
    for (int k = 0; k < in_stride; k++) {
      const dtype* in_ptr = x_data + n * in_channel + k;
      std::vector<std::pair<dtype, int>> vec;
      vec.resize(size);
      for (int i = 0; i < size; i++) {
        vec[i] = std::make_pair(in_ptr[i * in_stride], i);
      }
      // sort
      std::partial_sort(vec.begin(),
                        vec.begin() + 1,
                        vec.end(),
                        std::greater<std::pair<dtype, int>>());

      // out
      auto* out_ptr = output_data + n * out_channel + k;
      *out_ptr = vec[0].second;
    }
  }
}

TEST(argmax_arm, retrive_op) {
  auto argmax = KernelRegistry::Global().Create("arg_max");
  ASSERT_FALSE(argmax.empty());
  ASSERT_TRUE(argmax.front());
}

TEST(argmax_arm, init) {
  ArgmaxCompute argmax;
  ASSERT_EQ(argmax.precision(), PRECISION(kFloat));
  ASSERT_EQ(argmax.target(), TARGET(kARM));
}
TEST(argmax_arm, compute) {
  DeviceInfo::Init();
  for (auto n : {2, 3}) {
    for (auto c : {3, 4 /*, 128*/}) {
      for (auto h : {4, 5 /*, 56 , 112, 224, 512*/}) {
        for (auto w : {5, 6 /*, 56, 112, 224, 512*/}) {
          Tensor x;
          Tensor output;
          Tensor output_ref;
          int axis = (n + c + h + w) % 4;

          // get tensor x data
          x.Resize({n, c, h, w});
          auto* x_data = x.mutable_data<float>();
          for (int i = 0; i < x.dims().production(); i++) {
            float sign = i % 3 == 0 ? -1.0f : 1.0f;
            x_data[i] = sign * static_cast<float>(i % 128) * 0.013f;
          }

          // resize output and output_ref
          int nchw[] = {n, c, h, w};
          std::vector<int64_t> output_size(nchw, nchw + 4);
          output_size.erase(output_size.begin() + axis);
          output.Resize(output_size);
          output_ref.Resize(output_size);

          // obtain output_data
          ArgmaxCompute argmaxOp;
          std::unique_ptr<KernelContext> ctx(new KernelContext);
          ctx->As<ARMContext>();
          argmaxOp.SetContext(std::move(ctx));
          operators::ArgmaxParam param;
          param.X = &x;
          param.Out = &output;
          param.Axis = axis;
          argmaxOp.SetParam(param);
          argmaxOp.Launch();
          auto* output_data = output.mutable_data<int64_t>();

          // obtain output_ref_data
          param.Out = &output_ref;
          argmax_compute_ref<float>(param);
          auto* output_ref_data = output_ref.mutable_data<int64_t>();

          // compare
          for (int i = 0; i < output.dims().production(); i++) {
            EXPECT_NEAR(output_data[i], output_ref_data[i], 1e-5);
          }
        }
      }
    }
  }
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
USE_LITE_KERNEL(arg_max, kARM, kFloat, kNCHW, def);
