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
#include "lite/kernels/cuda/yolo_box_parser_compute.h"
// #include "lite/core/target_wrapper.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace cuda {

__global__ void yolo_tensor_bbox_num(const float * input, int * bbox_count, const uint gridSize, const uint numOutputClasses, const uint numBBoxes, float prob_thresh)
{
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
    uint z_id = blockIdx.z * blockDim.z + threadIdx.z;

    if ((x_id >= gridSize) || (y_id >= gridSize) || (z_id >= numBBoxes))
    {
        return;
    }

    const int numGridCells = gridSize * gridSize;
    const int bbindex = y_id * gridSize + x_id;

    // objectness
    float objectness = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)];
    if (objectness < prob_thresh)
    {
        return;
    }

    atomicAdd(bbox_count, 1);
}

inline __device__ void correct_yolo_box(float & x, float & y, float &w, float &h, float pic_w, float pic_h, float netw, float neth)
{
    int new_w = 0;
    int new_h = 0;
    if ((netw / pic_w) < (neth / pic_h))
    {
        new_w = netw;
        new_h = (pic_h * netw) / pic_w;
    }
    else
    {
        new_h = neth;
        new_w = (pic_w * neth) / pic_h;
    }
    
    x = (x - (netw - new_w) / 2.) / new_w;
    y = (y - (neth - new_h) / 2.) / new_h;
    w /= (float)new_w;
    h /= (float)new_h;
}

__global__ void yolo_tensor_parse_kernel(
    const float* input,
    const float* ImShape_data,
    const float* ScaleFactor_data,
    float* output,
    int * bbox_index,
    const uint gridSize,
    const uint numOutputClasses,
    const uint numBBoxes,
    const uint netw,
    const uint neth,
    float * biases,
    float prob_thresh)
{
    uint x_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint y_id = blockIdx.y * blockDim.y + threadIdx.y;
    uint z_id = blockIdx.z * blockDim.z + threadIdx.z;

    if ((x_id >= gridSize) || (y_id >= gridSize) || (z_id >= numBBoxes))
    {
        return;
    }

    const float pic_h = ImShape_data[0] / ScaleFactor_data[0];
    const float pic_w = ImShape_data[1] / ScaleFactor_data[1];

    const int numGridCells = gridSize * gridSize;
    const int bbindex = y_id * gridSize + x_id;

    // objectness
    float objectness = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 4)];
    if (objectness < prob_thresh)
    {
        return;
    }

    int cur_bbox_index = atomicAdd(bbox_index, 1);
    int tensor_index = cur_bbox_index * (5 + numOutputClasses);

    // x
    float x = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 0)];
    x = (float)((x + (float)x_id) * (float)netw) / (float)gridSize;
    
    // y
    float y = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 1)];
    y = (float)((y + (float)y_id) * (float)neth) / (float)gridSize;
    
    // w
    float w = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 2)];
    w = w * biases[2 * bias_index];
    
    // h
    float h = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + 3)];
    h = h * biases[2 * bias_index + 1];


    correct_yolo_box(x, y, w, h, pic_w, pic_h, netw, neth);

    output[tensor_index] = objectness;
    output[tensor_index + 1] = x;
    output[tensor_index + 2] = y;
    output[tensor_index + 3] = w;
    output[tensor_index + 4] = h;

    // Probabilities of classes 
    for (uint i = 0; i < numOutputClasses; ++i)
    {
        float prob = input[bbindex + numGridCells * (z_id * (5 + numOutputClasses) + (5 + i))] * objectness;
        output[tensor_index + 5 + i] = prob < prob_thresh ? 0. : prob;
    }

}

cudaError_t  yolo_tensor_parse_cuda(
    float* yolo_layer_tensor,       // [in] YOLO layer tensor input
    float* ImShape_data,
    float* ScaleFactor_data,
    float** bboxes_tensor_ptr,      // [out] Bounding boxes output tensor
    int * bbox_count_alloc_ptr,     // [in/out] maximum bounding box number allocated in device memory
    int * bbox_count_host,          // [in/out] bounding boxes number recorded in host memory
    int * bbox_count_device_ptr,        // [in/out] bounding boxes number calculated in device side
    int * bbox_index_device_ptr,        // [in] bounding box index for kernel threads shared access
    int gridSize,
    int numOutputClasses,
    int numBBoxes,
    int netw,
    int neth,
    float * biases_device,
    float prob_thresh, 
    cudaStream_t stream)
{
    dim3 threads_per_block(16, 16, 4);
    dim3 number_of_blocks((gridSize / threads_per_block.x) + 1,
        (gridSize / threads_per_block.y) + 1,
        (numBBoxes / threads_per_block.z) + 1);

    int bbox_count = 0;
    NV_CUDA_CHECK(cudaMemcpy(bbox_count_device_ptr, &bbox_count, sizeof(int), cudaMemcpyHostToDevice));
    yolo_tensor_bbox_num<<<number_of_blocks, threads_per_block, 0, stream>>> (
        yolo_layer_tensor, bbox_count_device_ptr, gridSize, numOutputClasses, numBBoxes, prob_thresh);
    NV_CUDA_CHECK(cudaMemcpy(&bbox_count, bbox_count_device_ptr, sizeof(int), cudaMemcpyDeviceToHost));

    //Record actual bbox number
    *bbox_count_host = bbox_count;  

    if (bbox_count <= 0)
    {
        cudaError_t status = cudaGetLastError();
        if (cudaSuccess != status)
        {
            printf("yolo_tensor_bbox_num error: %s\n", cudaGetErrorString(status));
        }
        return status;
    }

    // Obtain previous allocated bbox tensor in device side
    float* bbox_tensor = *bboxes_tensor_ptr;
    // Obtain previous maximum bbox number
    int bbox_count_max_alloc = *bbox_count_alloc_ptr;
    if (bbox_count > bbox_count_max_alloc)
    {
        printf("Bbox tensor expanded: %d -> %d!\n", bbox_count_max_alloc, bbox_count);
        cudaFree(bbox_tensor);
        cudaMalloc(&bbox_tensor, bbox_count * (5 + numOutputClasses) * sizeof(float));
        *bbox_count_alloc_ptr = bbox_count;
        *bboxes_tensor_ptr = bbox_tensor;
    }

    int bbox_index = 0;
    cudaMemcpy(bbox_index_device_ptr, &bbox_index, sizeof(int), cudaMemcpyHostToDevice);
    yolo_tensor_parse_kernel<<<number_of_blocks, threads_per_block, 0, stream>>>(
        yolo_layer_tensor, ImShape_data, ScaleFactor_data, 
        bbox_tensor, bbox_index_device_ptr, gridSize, numOutputClasses, numBBoxes, 
        netw, neth, biases_device, prob_thresh);

    cudaError_t status = cudaGetLastError();
    if (cudaSuccess != status)
    {
        printf("yolo_tensor_parse_kernel error: %s\n", cudaGetErrorString(status));
    }

    return status;
}

void YoloBoxParserCompute::Run() {
  auto& param = this->Param<param_t>();
  auto& ctx = this->ctx_->template As<CUDAContext>();
  auto stream = ctx.exec_stream();

  std::vector<lite::Tensor*> x = param.x;
  lite::Tensor* ImShape = param.ImShape;
  lite::Tensor* ScaleFactor = param.ScaleFactor;
  int batch = x[0]->dims[0];

  lite::Tensor* Boxes_Scores = param.Boxes_Scores;

  std::vector<int> anchors0 = param.anchors0;
  std::vector<int> anchors1 = param.anchors1;
  std::vector<int> anchors2 = param.anchors2;
  std::vector<int> downsample_ratio = param.downsample_ratio;
  int class_num = param.class_num;
  float conf_thresh = param.conf_thresh;
  bool clip_bbox = param.clip_bbox;
  float scale_x_y = param.scale_x_y;
  float bias = -0.5 * (scale_x_y - 1.);

  // maximum bbox counter in cpu memory
  int *bbox_count_alloc_ptr = (int *)malloc(sizeof(int));
  *bbox_count_alloc_ptr = 500;
  int bbox_count_host;
  // we pre-allocate 500 detection boxes in GPU memory
  float* bboxes_tensor_ptr;
  cudaMalloc((void**)bboxes_tensor_ptr, *bbox_count_alloc_ptr * (5 + class_num) sizeof(float));


 // box counter in gpu memory
 // box index counter in gpu memory
 int *bbox_count_device_ptr, *bbox_index_device_ptr;
 cudaMalloc((void**)&bbox_index_device_ptr, sizeof(int));
 cudaMalloc((void**)&bbox_count_device_ptr, sizeof(int));
 int *d_anchors0, *d_anchors1, *d_anchors2;
 cudaMalloc((void**)&d_anchors0, anchors0.size() * sizeof(int));
 cudaMalloc((void**)&d_anchors1, anchors1.size() * sizeof(int));
 cudaMalloc((void**)&d_anchors2, anchors2.size() * sizeof(int));
 float *dev_anchor_ptr[3];
 dev_anchor_ptr[0] = d_anchors0;
 dev_anchor_ptr[1] = d_anchors1;
 dev_anchor_ptr[2] = d_anchors2;
 cudaMemcpyAsync(d_anchors0, anchors0.data(), anchors0.size() * sizeof(int),cudaMemcpyHostToDevice, stream);
 cudaMemcpyAsync(d_anchors1, anchors1.data(), anchors0.size() * sizeof(int),cudaMemcpyHostToDevice, stream);
 cudaMemcpyAsync(d_anchors2, anchors2.data(), anchors0.size() * sizeof(int),cudaMemcpyHostToDevice, stream);

 for (int batch_index = 0; batch_index < batch; batch_index++)
 {
   for (int input_index = 0; input_index < x.size(); input_index++)
   {
     auto input_dims = x[input_index]->dims();
     const int grid_size = input_dims[2];
     const float* input = x[input_index]->data<float>() + batch_index * input_dims[1] * input_dims[2] * input_dims[3];
     const float* ImShape_data = ImShape->data<float>() + batch_index * 2;
     const float* ScaleFactor_data = ScaleFactor->data<float>() + batch_index * 2;
     const int neth = input_dims[2] * downsample_ratio[input_index];
     const int netw = input_dims[3] * downsample_ratio[input_index];
     yolo_tensor_parse_cuda(input,
                            ImShape_data,
                            ScaleFactor_data,
                            &bboxes_tensor_ptr, 
                            bbox_count_alloc_ptr,
                            &bbox_count_host,
                            bbox_count_device_ptr,
                            bbox_index_device_ptr,
                            input_dims[2],
                            class_num,
                            3,
                            netw,
                            neth,
                            dev_anchor_ptr[input_index],
                            stream
                            );
   }
 }
 cudaError_t error = cudaGetLastError();
 if (error != cudaSuccess) LOG(INFO) << cudaGetErrorString(error);
}

}  // namespace cuda
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(yolo_box_parser,
                     kCUDA,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::cuda:YoloBoxParserCompute,
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
