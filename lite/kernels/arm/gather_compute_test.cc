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

#include "lite/kernels/arm/gather_compute.h"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <vector>
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void GatherComputeRef(const operators::GatherParam& param) {
  auto* p_output = param.Out->mutable_data<float>();
  auto index_size = param.Index->dims()[0];
  auto src_dims = param.X->dims();
  const float* p_src = param.X->data<float>();
  const float* p_index = param.Index->data<float>();

  int slice_size = 1;
  for (int i = 1; i < src_dims.size(); ++i) {
    slice_size *= src_dims[i];
  }
  for (int i = 0; i < index_size; ++i) {
    int index_ = p_index[i];
    memcpy(p_output + i * slice_size,
           p_src + index_ * slice_size,
           slice_size * sizeof(float));
  }
}

TEST(gather_arm, init) {
  GatherCompute gather;
  ASSERT_EQ(gather.precision(), PRECISION(kFloat));
  ASSERT_EQ(gather.target(), TARGET(kARM));
}

TEST(gather_arm, compute) {
  GatherCompute gather;
  operators::GatherParam param;

  lite::Tensor x;
  lite::Tensor output;
  lite::Tensor index;
  lite::Tensor output_ref;

  for (auto n : {1}) {
    for (auto c : {1, 3, 5}) {
      for (auto h : {3, 16, 20, 32}) {
        for (auto w : {3, 16, 20, 32}) {
            auto dims = DDim(std::vector<int64_t>({n, c, h, w}));
            x.Resize(dims);
            index.Resize(DDim(std::vector<int64_t>({0, 1, 1, 1})));
            output.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
            output_ref.Resize(DDim(std::vector<int64_t>({n, c, h, w})));
            
            auto* x_data = x.mutable_data<float>();
            auto* output_data = output.mutable_data<float>();
             auto* output_data_ref = output_ref.mutable_data<float>();
            
            for (int i = 0; i < x.dims().production(); i++) {
                x_data[i] = i % 255 * 0.001;
            }
            param.X = &x;
            param.Out = &output;
            param.Index = &index;
                
            gather.SetParam(param);
            gather.Run();

            param.Out = &output_ref;
            GatherComputeRef(param);
            for (int i = 0; i < output.dims().production(); i++) {
                EXPECT_NEAR(output_data[i], output_data_ref[i], 1e-4);
            }
          }
        }
      }
  }
}

TEST(gather, retrive_op) {
  auto gather =
      KernelRegistry::Global().Create<TARGET(kARM), PRECISION(kFloat)>(
          "gather");
  ASSERT_FALSE(gather.empty());
  ASSERT_TRUE(gather.front());
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(gather, kARM, kFloat, kNCHW, def);
