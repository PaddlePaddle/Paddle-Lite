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

#include "lite/kernels/cuda/sequence_topk_avg_pooling_compute.h"
#include <gtest/gtest.h>
#include <memory>
#include <utility>
#include <vector>

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

static void get_topk(std::vector<float>* src,
                     int top_k,
                     int real_k,
                     float* dst) {
  for (int k = 0; k < real_k; k++) {
    float max_data = -1e10;
    int max_index = -1;
    for (unsigned int i = 0; i < (*src).size(); i++) {
      if (max_data < (*src)[i]) {
        max_index = i;
        max_data = (*src)[i];
      }
    }
    (*src)[max_index] = -1e10;
    dst[k] = max_data;
  }
  for (int k = real_k; k < top_k; k++) {
    dst[k] = static_cast<float>(0.f);
  }
}

static void topk_avg_pooling_basic(const lite::Tensor* X,
                                   const lite::Tensor* ROW,
                                   const lite::Tensor* COLUMN,
                                   int channel_num,
                                   std::vector<int> topks,
                                   lite::Tensor* Out) {
  Out->set_lod(X->lod());
  auto height_offset = ROW->lod()[0];
  auto width_offset = COLUMN->lod()[0];

  const float* input_data = X->data<float>();

  DDim X_dims = X->dims();
  int num = X_dims[0];
  int channel = X_dims[1];
  int height_stride = X_dims[2];
  int width_stride = X_dims[3];
  CHECK_EQ(channel_num, channel) << "feat map num is not valid";
  int dim0 = 0;
  dim0 = ROW->dims()[0];
  CHECK_EQ(dim0, height_offset[height_offset.size() - 1]);
  Out->set_lod(ROW->lod());
  int num_k = topks.size();
  int max_k = topks[num_k - 1];
  auto offset = X->lod()[0];
  int64_t temp_index = offset[offset.size() - 1];
  std::vector<int64_t> output_shape{temp_index, channel * num_k, 1, 1};
  Out->Resize(lite::DDim(output_shape));
  float* output_data = Out->mutable_data<float>();

  for (int i = 0; i < num; i++) {
    int height = height_offset[i + 1] - height_offset[i];
    int width = width_offset[i + 1] - width_offset[i];
    std::vector<float> vec;
    std::vector<float> topk_value;
    std::vector<float> sum;
    topk_value.resize(max_k);
    sum.resize(max_k);
    int real_k = max_k < width ? max_k : width;
    for (int h = 0; h < height; h++) {
      for (int c = 0; c < channel; c++) {
        auto tmp_in_data =
            input_data + ((i * channel + c) * height_stride + h) * width_stride;
        auto tmp_out_data =
            output_data + ((height_offset[i] + h) * channel + c) * num_k;

        vec.clear();
        for (int w = 0; w < width; w++) {
          vec.push_back(tmp_in_data[w]);
        }
        get_topk(&vec, max_k, real_k, &topk_value[0]);
        sum[0] = topk_value[0];
        for (int m = 1; m < max_k; m++) {
          sum[m] = sum[m - 1] + topk_value[m];
        }
        for (unsigned int m = 0; m < topks.size(); m++) {
          tmp_out_data[m] = sum[topks[m] - 1] / topks[m];
        }
      }
    }
  }
}

TEST(sequence_topk_avg_pooling, normal) {
  SequenceTopkAvgPoolingCompute<float> seq_topk_kernel;
  std::unique_ptr<KernelContext> ctx(new KernelContext);
  auto& context = ctx->As<CUDAContext>();
  operators::SequenceTopkAvgPoolingParam param;

  lite::Tensor X, X_gpu, ROW, ROW_gpu, COLUMN, COLUMN_gpu;
  lite::Tensor Out, Out_cpu, out_ref, pos, pos_cpu, pos_ref;

  lite::DDim shape(std::vector<int64_t>{2, 3, 7, 8});
  std::vector<int> topks = {1, 2, 3, 4, 5};
  int input_channel = 3;
  std::vector<std::vector<uint64_t>> height_seq_offset;
  std::vector<std::vector<uint64_t>> width_seq_offset;
  height_seq_offset.resize(1);
  width_seq_offset.resize(1);
  int cumsum_width = 0;
  int cumsum_height = 0;
  height_seq_offset[0].push_back(cumsum_height);
  width_seq_offset[0].push_back(cumsum_width);
  for (int i = 0; i < shape[0]; i++) {
    int cur_width = std::rand() % shape[3] + 1;
    int cur_height = shape[2];
    cumsum_width += cur_width;
    cumsum_height += cur_height;
    height_seq_offset[0].push_back(cumsum_height);
    width_seq_offset[0].push_back(cumsum_width);
  }
  lite::DDim shape_1(std::vector<int64_t>{cumsum_height, 10, 1, 1});
  lite::DDim shape_2(std::vector<int64_t>{cumsum_width, 10, 1, 1});
  ROW.Resize(shape_1);
  auto* row_data = ROW.mutable_data<float>();
  for (int i = 0; i < ROW.numel(); ++i) {
    row_data[i] = i;
  }

  COLUMN.Resize(shape_2);
  auto* column_data = COLUMN.mutable_data<float>();
  for (int i = 0; i < COLUMN.numel(); ++i) {
    column_data[i] = i;
  }
  LoD width_lod;
  width_lod.push_back(width_seq_offset[0]);
  LoD height_lod;
  height_lod.push_back(height_seq_offset[0]);

  std::vector<uint64_t> x_lod_vec;
  x_lod_vec.push_back(0);
  for (size_t i = 1; i < width_seq_offset[0].size(); ++i) {
    int height = width_seq_offset[0][i] - width_seq_offset[0][i - 1];
    int width = height_seq_offset[0][i] - height_seq_offset[0][i - 1];
    x_lod_vec.push_back(height * width * input_channel + x_lod_vec[i - 1]);
  }
  LoD x_lod;
  x_lod.push_back(x_lod_vec);
  X.set_lod(x_lod);
  ROW.set_lod(height_lod);
  COLUMN.set_lod(width_lod);

  X.Resize(shape);
  auto* x_data = X.mutable_data<float>();
  for (int i = 0; i < X.numel(); ++i) {
    x_data[i] = (-1) ^ i;
  }

  auto row_dim = ROW.dims();
  auto num_k = topks.size();
  auto row_shape_0 = row_dim[0];
  std::vector<int64_t> vec_out_shape;
  vec_out_shape.push_back(row_shape_0);
  vec_out_shape.push_back(input_channel * num_k);

  Out.Resize(lite::DDim(vec_out_shape));

  topk_avg_pooling_basic(&X, &ROW, &COLUMN, input_channel, topks, &out_ref);
  X_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(x_data, X.dims());
  ROW_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(row_data, ROW.dims());
  COLUMN_gpu.Assign<float, lite::DDim, TARGET(kCUDA)>(column_data,
                                                      COLUMN.dims());

  param.X = &X_gpu;
  param.ROW = &ROW_gpu;
  param.COLUMN = &COLUMN_gpu;
  param.channel_num = input_channel;
  param.topks = topks;
  param.Out = &Out;
  param.pos = &pos;
  seq_topk_kernel.SetParam(param);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  context.SetExecStream(stream);
  seq_topk_kernel.SetContext(std::move(ctx));
  seq_topk_kernel.Run();

  cudaDeviceSynchronize();

  const float* out_data = Out.data<float>();
  float* out_cpu_data = Out_cpu.mutable_data<float>();
  CopySync<TARGET(kCUDA)>(
      out_cpu_data, out_data, sizeof(float) * Out.numel(), IoDirection::DtoH);

  for (int i = 0; i < Out.numel(); ++i) {
    EXPECT_NEAR(out_cpu_data[i], out_ref.data<float>()[i], 1e-5);
  }
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
