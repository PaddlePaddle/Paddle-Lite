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
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/io_copy_compute.h"

namespace paddle {
namespace lite {

TEST(tensor, test) {
  TensorLite tensor;
  DDimLite ddim({1, 8});
  tensor.Resize(ddim);

  for (int i = 0; i < 8; i++) {
    tensor.mutable_data<int>()[i] = i;
  }
}

#ifdef LITE_WITH_OPENCL
TEST(tensor, test_ocl_image2d) {
  using DTYPE = float;
  const size_t N = 1;
  const size_t C = 3;
  const size_t H = 5;
  const size_t W = 7;

  TensorLite x;
  DDimLite x_dims = DDim(std::vector<int64_t>({N, C, H, W}));
  x.Resize(x_dims);
  DTYPE *x_data = x.mutable_data<DTYPE>();
  for (int eidx = 0; eidx < x_dims.production(); ++eidx) {
    x_data[eidx] = eidx;
  }

  LOG(INFO) << "x.dims().size():" << x.dims().size();
  for (size_t dim_idx = 0; dim_idx < x.dims().size(); ++dim_idx) {
    LOG(INFO) << "x.dims()[" << dim_idx << "]:" << x.dims()[dim_idx];
  }

// io_copy: from cpu to gpu buffer
//     io_copy kernel called CopyFromHostSync(void* target, const void* source,
//     size_t size);
//     CopyFromHostSync called TargetWrapperCL::MemcpySync(target, source, size,
//     IoDirection::HtoD);
// void* x_data_gpu = nullptr;
// CopyFromHostSync(x_data_gpu, x_data, x_dims.production() * sizeof(DTYPE));
// ref:/lite/kernels/opencl/io_copy_compute_test.cc

// data_layout_trans(cl:Buffer -> cl:Image2D): from nchw to image2d

#if 0
  x.mutable_data<DTYPE, cl::Image2D>();
  std::array<size_t, 2> image2d_shape{0, 0};
  std::array<size_t, 2> image2d_pitch{0, 0};
  x.image2d_shape(&image2d_shape, &image2d_pitch);
  LOG(INFO) << "image2d_shape['w']:" << image2d_shape[0];
  LOG(INFO) << "image2d_shape['h']:" << image2d_shape[1];
  LOG(INFO) << "image2d_pitch['row']:" << image2d_pitch[0];
  LOG(INFO) << "image2d_pitch['slice']:" << image2d_pitch[1];
#endif
}
#endif

}  // namespace lite
}  // namespace paddle
