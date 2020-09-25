/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <vector>
#include "lite/core/op_registry.h"
#include "lite/kernels/cuda/yolo_box_compute.h"
// #include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

__host__ __device__ inline int GetEntryIndex(int batch,
                                             int an_idx,
                                             int hw_idx,
                                             int an_num,
                                             int an_stride,
                                             int stride,
                                             int entry) {
  return (batch * an_num + an_idx) * an_stride + entry * stride + hw_idx;
}

template <typename T>
__host__ __device__ inline T sigmoid(T x) {
  return 1.0 / (1.0 + std::exp(-x));
}

template <typename T>
__host__ __device__ inline void GetYoloBox(T* box,
                                           const T* x,
                                           const int* anchors,
                                           int i,
                                           int j,
                                           int an_idx,
                                           int grid_size,
                                           int input_size,
                                           int index,
                                           int stride,
                                           int img_height,
                                           int img_width,
                                           float scale,
                                           float bias) {
  box[0] = (i + sigmoid<T>(x[index]) * scale + bias) * img_width / grid_size;
  box[1] = (j + sigmoid<T>(x[index + stride]) * scale + bias) * img_height /
           grid_size;
  box[2] = std::exp(x[index + 2 * stride]) * anchors[2 * an_idx] * img_width /
           input_size;
  box[3] = std::exp(x[index + 3 * stride]) * anchors[2 * an_idx + 1] *
           img_height / input_size;
}

template <typename T>
__host__ __device__ inline void CalcDetectionBox(T* boxes,
                                                 T* box,
                                                 const int box_idx,
                                                 const int img_height,
                                                 const int img_width,
                                                 bool clip_bbox) {
  boxes[box_idx] = box[0] - box[2] / 2;
  boxes[box_idx + 1] = box[1] - box[3] / 2;
  boxes[box_idx + 2] = box[0] + box[2] / 2;
  boxes[box_idx + 3] = box[1] + box[3] / 2;

  if (!clip_bbox) {
    return;
  }
  boxes[box_idx] = boxes[box_idx] > 0 ? boxes[box_idx] : static_cast<T>(0);
  boxes[box_idx + 1] =
      boxes[box_idx + 1] > 0 ? boxes[box_idx + 1] : static_cast<T>(0);
  boxes[box_idx + 2] = boxes[box_idx + 2] < img_width - 1
                           ? boxes[box_idx + 2]
                           : static_cast<T>(img_width - 1);
  boxes[box_idx + 3] = boxes[box_idx + 3] < img_height - 1
                           ? boxes[box_idx + 3]
                           : static_cast<T>(img_height - 1);
}

template <typename T>
__host__ __device__ inline void CalcLabelScore(T* scores,
                                               const T* input,
                                               const int label_idx,
                                               const int score_idx,
                                               const int class_num,
                                               const T conf,
                                               const int stride) {
  for (int i = 0; i < class_num; ++i) {
    scores[score_idx + i] = conf * sigmoid<T>(input[label_idx + i * stride]);
  }
}

template <typename T>
__global__ void KeYoloBoxFw(const T* input,
                            const int* imgsize,
                            T* boxes,
                            T* scores,
                            const float conf_thresh,
                            const int* anchors,
                            const int n,
                            const int h,
                            const int w,
                            const int an_num,
                            const int class_num,
                            const int box_num,
                            int input_size,
                            bool clip_bbox,
                            float scale,
                            float bias) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  T box[4];
  for (; tid < n * box_num; tid += stride) {
    int grid_num = h * w;
    int i = tid / box_num;
    int j = (tid % box_num) / grid_num;
    int k = (tid % grid_num) / w;
    int l = tid % w;

    int an_stride = (5 + class_num) * grid_num;
    int img_height = imgsize[2 * i];
    int img_width = imgsize[2 * i + 1];

    int obj_idx =
        GetEntryIndex(i, j, k * w + l, an_num, an_stride, grid_num, 4);
    T conf = sigmoid<T>(input[obj_idx]);
    if (conf < conf_thresh) {
      continue;
    }

    int box_idx =
        GetEntryIndex(i, j, k * w + l, an_num, an_stride, grid_num, 0);
    GetYoloBox<T>(box,
                  input,
                  anchors,
                  l,
                  k,
                  j,
                  h,
                  input_size,
                  box_idx,
                  grid_num,
                  img_height,
                  img_width,
                  scale,
                  bias);
    box_idx = (i * box_num + j * grid_num + k * w + l) * 4;
    CalcDetectionBox<T>(boxes, box, box_idx, img_height, img_width, clip_bbox);

    int label_idx =
        GetEntryIndex(i, j, k * w + l, an_num, an_stride, grid_num, 5);
    int score_idx = (i * box_num + j * grid_num + k * w + l) * class_num;
    CalcLabelScore<T>(
        scores, input, label_idx, score_idx, class_num, conf, grid_num);
  }
}

void YoloBoxCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  lite::Tensor* X = param.X;
  lite::Tensor* ImgSize = param.ImgSize;
  lite::Tensor* Boxes = param.Boxes;
  lite::Tensor* Scores = param.Scores;
  std::vector<int> anchors = param.anchors;
  int class_num = param.class_num;
  float conf_thresh = param.conf_thresh;
  int downsample_ratio = param.downsample_ratio;
  bool clip_bbox = param.clip_bbox;
  float scale_x_y = param.scale_x_y;
  float bias = -0.5 * (scale_x_y - 1.);

  const float* input = X->data<float>();
  const int* imgsize = ImgSize->data<int>();
  float* boxes = Boxes->mutable_data<float>(TARGET(kCUDA));
  float* scores = Scores->mutable_data<float>(TARGET(kCUDA));
  TargetWrapperCuda::MemsetAsync(
      boxes, 0, Boxes->numel() * sizeof(float), stream);
  TargetWrapperCuda::MemsetAsync(
      scores, 0, Scores->numel() * sizeof(float), stream);

  const int n = X->dims()[0];
  const int h = X->dims()[2];
  const int w = X->dims()[3];
  const int box_num = Boxes->dims()[1];
  const int an_num = anchors.size() / 2;
  int input_size = downsample_ratio * h;

  anchors_.Resize({static_cast<int64_t>(anchors.size())});
  int* d_anchors = anchors_.mutable_data<int>(TARGET(kCUDA));
  TargetWrapperCuda::MemcpyAsync(d_anchors,
                                 anchors.data(),
                                 sizeof(int) * anchors.size(),
                                 IoDirection::HtoD,
                                 stream);

  int threads = 512;
  int blocks = (n * box_num + threads - 1) / threads;
  blocks = blocks > 8 ? 8 : blocks;

  KeYoloBoxFw<float><<<blocks, threads, 0, stream>>>(input,
                                                     imgsize,
                                                     boxes,
                                                     scores,
                                                     conf_thresh,
                                                     d_anchors,
                                                     n,
                                                     h,
                                                     w,
                                                     an_num,
                                                     class_num,
                                                     box_num,
                                                     input_size,
                                                     clip_bbox,
                                                     scale_x_y,
                                                     bias);
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(yolo_box,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda::YoloBoxCompute,
                     def)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kFloat),
                                      DATALAYOUT(kNCHW))})
    .BindInput("ImgSize",
               {LiteType::GetTensorTy(TARGET(kCUDA),
                                      PRECISION(kInt32),
                                      DATALAYOUT(kNCHW))})
    .BindOutput("Boxes",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .BindOutput("Scores",
                {LiteType::GetTensorTy(TARGET(kCUDA),
                                       PRECISION(kFloat),
                                       DATALAYOUT(kNCHW))})
    .Finalize();
