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

#include <memory>
#include <vector>

#include "CL/cl.h"

#include "framework/cl/cl_deleter.h"
#include "framework/cl/cl_engine.h"
#include "framework/cl/cl_half.h"
#include "framework/cl/cl_image_converter.h"
#include "framework/cl/cl_tool.h"
#include "framework/ddim.h"
#include "framework/tensor.h"

namespace paddle_mobile {
namespace framework {

class CLImage {
 public:
  CLImage() = default;

  ~CLImage() {
    if (tensor_data_ != nullptr) {
      delete[](tensor_data_);
    }

    if (image_converter_) {
      delete (image_converter_);
    }
  }
  /*
   * will not hold input tensor data, memcpy in this method
   * */
  void SetTensorData(float *tensorData, const DDim &dim) {
    int numel = product(dim);
    if (tensor_data_ != nullptr) {
      delete[](tensor_data_);
      tensor_data_ = nullptr;
    }
    tensor_data_ = new float[numel];
    memcpy(tensor_data_, tensorData, numel * sizeof(float));
    tensor_dims_ = dim;
  }

  bool isInit() { return initialized_; }
  /*
   * need call SetTensorData first
   *
   * folder when one dim or two dim
   * */
  void InitCLImage(cl_context context, cl_command_queue command_queue) {
    PADDLE_MOBILE_ENFORCE(tensor_data_ != nullptr,
                          " need call SetTensorData first");
    CLImageConverterFolder *folder_converter = new CLImageConverterFolder();
    InitCLImage(context, command_queue, folder_converter);
  }

  void InitNormalCLImage(cl_context context, cl_command_queue command_queue) {
    PADDLE_MOBILE_ENFORCE(tensor_data_ != nullptr,
                          " need call SetTensorData first");
    CLImageConverterNormal *normal_converter = new CLImageConverterNormal();
    InitCLImage(context, command_queue, normal_converter);
  }

  void InitCLImage(cl_context context, cl_command_queue command_queue,
                   CLImageConverterBase *converter) {
    if (image_converter_ != nullptr) {
      delete (image_converter_);
    }

    PADDLE_MOBILE_ENFORCE(tensor_data_ != nullptr,
                          " need call SetTensorData first");

    DLOG << " begin init cl image ";
    image_dims_ = converter->InitImageDimInfoWith(tensor_dims_);

    half_t *image_data = new half_t[product(image_dims_) * 4];

    DLOG << " convert to image";
    converter->NCHWToImage(tensor_data_, image_data, tensor_dims_);
    DLOG << " end convert to image";

    InitCLImage(context, image_dims_[0], image_dims_[1], image_data);

    delete[](image_data);
    delete[](tensor_data_);

    command_queue_ = command_queue;
    tensor_data_ = nullptr;
    image_converter_ = converter;
    initialized_ = true;
    DLOG << " end init cl image";
  }

  void InitNImage(cl_context context, cl_command_queue command_queue) {
    if (tensor_data_ == nullptr) {
      PADDLE_MOBILE_THROW_EXCEPTION(" need call SetTensorData first");
    }
    CLImageConverterNWBlock *folder_converter = new CLImageConverterNWBlock();
    InitCLImage(context, command_queue, folder_converter);
    PADDLE_MOBILE_ENFORCE(tensor_dims_.size() == 4, " tensor dim is not 4");
  }
  void InitDWImage(cl_context context, cl_command_queue command_queue) {
    if (tensor_data_ == nullptr) {
      PADDLE_MOBILE_THROW_EXCEPTION(" need call SetTensorData first");
    }
    CLImageConverterDWBlock *dw_converter = new CLImageConverterDWBlock();
    InitCLImage(context, command_queue, dw_converter);
    PADDLE_MOBILE_ENFORCE(tensor_dims_.size() == 4, " tensor dim is not 4");
  }

  void InitEmptyImage(cl_context context, cl_command_queue command_queue,
                      const DDim &dim) {
    PADDLE_MOBILE_ENFORCE(tensor_data_ == nullptr,
                          " empty image tensor data shouldn't have value");

    //    CLImageConverterFolder *folder_converter = new
    //    CLImageConverterFolder();
    CLImageConverterNormal *normal_converter = new CLImageConverterNormal();

    DLOG << " to get image dims ";
    image_dims_ = normal_converter->InitImageDimInfoWith(dim);
    DLOG << " end get image dims " << image_dims_;

    InitCLImage(context, image_dims_[0], image_dims_[1], nullptr);

    tensor_dims_ = dim;
    command_queue_ = command_queue;
    image_converter_ = normal_converter;
    cl_event_ = CLEngine::Instance()->CreateEvent(context);
    initialized_ = true;
    DLOG << " end init cl image";
  }
  // create fake size cl_mem for mem share
  void InitFakeSizeImage(cl_context context, cl_command_queue command_queue,
                         const DDim &need_dims, const DDim &real_dims) {
    PADDLE_MOBILE_ENFORCE(tensor_data_ == nullptr,
                          " empty image tensor data shouldn't have value");

    CLImageConverterNormal *normal_converter = new CLImageConverterNormal();

    real_image_dims = normal_converter->InitImageDimInfoWith(real_dims);
    real_tensor_dims = real_dims;

    image_dims_ = normal_converter->InitImageDimInfoWith(need_dims);
    InitCLImage(context, image_dims_[0], image_dims_[1], nullptr);

    tensor_dims_ = need_dims;
    command_queue_ = command_queue;
    image_converter_ = normal_converter;
    cl_event_ = CLEngine::Instance()->CreateEvent(context);
    initialized_ = true;
    DLOG << " end init cl image";
  }

  void InitWithExitedMem(cl_context context, cl_command_queue command_queue,
                         DDim need_dims, const CLImage &src) {
    CLImageConverterNormal *normal_converter = new CLImageConverterNormal();

    real_image_dims = normal_converter->InitImageDimInfoWith(src.dims());
    real_tensor_dims = src.dims();

    image_dims_ = normal_converter->InitImageDimInfoWith(need_dims);
    // InitCLImage(context, image_dims_[0], image_dims_[1], nullptr);
    if (cl_image_ != src.cl_image_) {
      cl_image_.reset(src.cl_image_.get());
    }

    tensor_dims_ = need_dims;
    command_queue_ = command_queue;
    image_converter_ = normal_converter;
    cl_event_ = CLEngine::Instance()->CreateEvent(context);
    initialized_ = true;
    DLOG << " end init cl image";
  }

  void InitConv2dTransposeFilterCLImage(cl_context context,
                                        cl_command_queue command_queue) {
    PADDLE_MOBILE_ENFORCE(tensor_data_ != nullptr,
                          " need call SetTensorData first");
    CLImageConverterConv2dTransposeTransWeight *converter =
        new CLImageConverterConv2dTransposeTransWeight();
    InitCLImage(context, command_queue, converter);
  }

  /*! The internal of two tensors share the same memory block. */
  inline CLImage &ShareHolderWith(const CLImage &src) {
    PADDLE_MOBILE_ENFORCE(
        src.cl_image_ != nullptr,
        "Tensor holds no memory. Call Tensor::mutable_data first.")

    if (cl_image_ != src.cl_image_) {
      cl_image_.reset(src.cl_image_.get());
    }
    return *this;
  }

  cl_mem GetCLImage() const { return cl_image_.get(); }

  const DDim &ImageDims() const { return image_dims_; }

  inline size_t ImageWidth() const { return image_dims_[0]; }

  inline size_t ImageHeight() const { return image_dims_[1]; }

  inline cl_command_queue CommandQueue() const { return command_queue_; }

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

  cl_event GetClEvent() const { return cl_event_.get(); }

  CLImageConverterBase *Converter() const { return image_converter_; }

 private:
  void InitCLImage(cl_context context, int width, int height, void *data) {
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
    cl_int err;
    cl_mem cl_image = clCreateImage(
        context, CL_MEM_READ_WRITE | (data ? CL_MEM_COPY_HOST_PTR : 0),
        &cf,   // const cl_image_format *image_format
        &cid,  // const cl_image_desc *image_desc
        data,  // void *host_ptr
        &err);
    cl_image_.reset(cl_image);
    if (err != CL_SUCCESS) {
      CL_CHECK_ERRORS(err);
      PADDLE_MOBILE_THROW_EXCEPTION(" create image 2d error ");
    }
  }

  bool initialized_ = false;
  std::unique_ptr<_cl_mem, CLMemDeleter> cl_image_;
  std::unique_ptr<_cl_event, CLEventDeleter> cl_event_;
  DDim tensor_dims_;
  DDim image_dims_;
  // real image dims usually it is same as image_dims
  DDim real_image_dims;
  // real tensor dims usually it is same as tensor dims
  DDim real_tensor_dims;
  float *tensor_data_ = nullptr;
  cl_context context_;
  cl_command_queue command_queue_;
  CLImageConverterBase *image_converter_ = nullptr;
};

void TensorToCLImage(Tensor *tensor, CLImage *image, cl_context context,
                     cl_command_queue commandQueue, cl_kernel kernel);

void CLImageToTensor(CLImage *image, Tensor *tensor, cl_context context,
                     cl_command_queue commandQueue, cl_kernel kernel);

#ifdef PADDLE_MOBILE_DEBUG
Print &operator<<(Print &printer, const CLImage &image);
#endif

}  // namespace framework
}  // namespace paddle_mobile
