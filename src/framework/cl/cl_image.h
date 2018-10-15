/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>

#include "CL/cl.h"

#include "framework/cl/cl_half.h"
#include "framework/cl/cl_tool.h"
#include "framework/ddim.h"
#include "framework/tensor.h"

namespace paddle_mobile {
namespace framework {

class CLImage {
 public:
  CLImage() = default;

  /*
   * will not hold input tensor data, memcpy in this method
   * */
  void SetTensorData(float *tensorData, const DDim &dim) {
    int numel = product(dim);
    if (tensor_data_ != nullptr) {
      delete[](tensor_data_);
    }
    tensor_data_ = new float[numel];
    memcpy(tensor_data_, tensorData, numel);
    tensor_dims_ = dim;
  }

  /*
   * need call SetTensorData first
   * */
  void InitCLImage(cl_context context) {
    if (tensor_data_ == nullptr) {
      PADDLE_MOBILE_THROW_EXCEPTION(" need call SetTensorData first");
    }
    InitCLImage(context, tensor_data_, tensor_dims_);
    delete[](tensor_data_);
    tensor_data_ = nullptr;
    initialized_ = true;
  }

  void InitEmptyImage(cl_context context, const DDim &dim) {
    if (tensor_data_ != nullptr) {
      PADDLE_MOBILE_THROW_EXCEPTION(
          " empty image tensor data shouldn't have value");
    }
    DLOG << " init empty image ";
    InitCLImage(context, nullptr, dim);
    initialized_ = true;
  }

  cl_mem GetCLImage() const { return cl_image_; }

  const DDim &ImageDims() { return image_dims_; }

  inline size_t ImageWidth() const { return image_width_; }

  inline size_t ImageHeight() const { return image_height_; }

  /*
   * block of channels, 4 channel one block
   * */
  inline size_t CBlock() const { return c_block_; }

  /*
   *  width of original tensor
   * */
  inline size_t WidthOfOneBlock() const { return width_of_one_block_; }

  /*
   *  height of original tensor
   * */
  inline size_t HeightOfOneBlock() const { return height_of_one_block_; }

  /*
   *  resize original tensor dim
   * */
  inline CLImage &Resize(const DDim &dims) {
    tensor_dims_ = dims;
    return *this;
  }

  template <typename T>
  T *data() const {
    if (initialized_) {
      PADDLE_MOBILE_THROW_EXCEPTION(
          " cl image has initialized, tensor data has been deleted, can't use "
          "tensor data");
    }
    return reinterpret_cast<T *>(tensor_data_);
  }

  /*
   *  numel of tensor dim
   * */
  inline int64_t numel() const { return product(tensor_dims_); }

  /*
   *  original tensor dim
   * */
  const DDim &dims() const { return tensor_dims_; }

 private:
  void InitCLImage(cl_context context, float *tensor_data, const DDim &dim) {
    DLOG << " tensor dim: " << dim;
    // NCHW -> [W * (C+3)/4, H * N]
    tensor_dims_ = dim;
    if (tensor_data) {
      tensor_data_ = tensor_data;
    } else {
      int numel = 1;
      for (int i = 0; i < dim.size(); i++) {
        numel *= dim[i];
      }
      tensor_data_ = static_cast<float *>(
          paddle_mobile::memory::Alloc(sizeof(float) * numel));
      for (int i = 0; i < numel; i++) {
        tensor_data_[i] = 0;
      }
    }

    size_t new_dims[] = {1, 1, 1, 1};

    for (int j = 0; j < dim.size(); ++j) {
      new_dims[4 - dim.size() + j] = dim[j];
    }

    size_t N, C, H, W;

    N = new_dims[0];
    C = new_dims[1];
    H = new_dims[2];
    W = new_dims[3];

    width_of_one_block_ = W;
    height_of_one_block_ = H;

    size_t width = W * ((C + 3) / 4);
    size_t height = H * N;

    image_width_ = width;
    image_height_ = height;
    image_dims_ = make_ddim({image_width_, image_height_});

    std::unique_ptr<half_t[]> imageData{};
    int count = 0;
    if (tensor_data != nullptr) {
      imageData.reset(new half_t[width * height * 4]);
      float *p = tensor_data;
      size_t i0 = 0;
      for (int n = 0; n < N; n++) {
        for (int c = 0; c < C; c++) {
          size_t i1 = i0;
          for (int h = 0; h < H; h++) {
            size_t i2 = (i1 << 2) + c % 4;
            for (int w = 0; w < W; w++) {
              if (i2 >= width * height * 4) {
                printf("%d > %d ----> %d, %d, %d, %d --- %d, %d, %d\n", i2,
                       width * height * 4, n, c, h, w, i0, i1, i2);
              }
              assert(i2 < width * height * 4);

              imageData[i2] = Float2Half(*p);
              i2 += 4;
              p++;
              //              count++;
              //              DLOG<<count;
            }
            i1 += width;
          }
        }
        i0 += width * H;
      }
    }
    cl_int err;
    DLOG << " image width: " << width;
    DLOG << " image height: " << height;

    cl_image_format cf = {.image_channel_order = CL_RGBA,
                          .image_channel_data_type = CL_HALF_FLOAT};
    cl_image_desc cid = {
        .image_type = CL_MEM_OBJECT_IMAGE2D,
        .image_width = width,
        .image_height = height,
        .image_depth = 1,
        .image_array_size = 1,
        .image_row_pitch = 0,
        .image_slice_pitch = 0,
        .num_mip_levels = 0,
        .num_samples = 0,
        // .buffer = nullptr
    };
    cid.buffer = nullptr;
    cl_image_ = clCreateImage(
        context, CL_MEM_READ_WRITE | (imageData ? CL_MEM_COPY_HOST_PTR : 0),
        &cf,   // const cl_image_format *image_format
        &cid,  // const cl_image_desc *image_desc
        reinterpret_cast<void *>(imageData.get()),  // void *host_ptr
        &err);

    if (err != CL_SUCCESS) {
      CL_CHECK_ERRORS(err);
      PADDLE_MOBILE_THROW_EXCEPTION(" create image 2d error ");
    }
  }

  bool initialized_ = false;
  cl_mem cl_image_;
  size_t image_width_;
  size_t width_of_one_block_;
  size_t height_of_one_block_;
  size_t image_height_;
  size_t c_block_;
  DDim tensor_dims_;
  DDim image_dims_;
  float *tensor_data_;
  cl_context context_;
};

void TensorToCLImage(Tensor *tensor, CLImage *image,
                     cl_command_queue commandQueue);

void CLImageToTensor(CLImage *image, Tensor *tensor,
                     cl_command_queue commandQueue);

#ifdef PADDLE_MOBILE_DEBUG
Print &operator<<(Print &printer, const CLImage &image);
#endif

}  // namespace framework
}  // namespace paddle_mobile
