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

#include "lite/backends/opencl/target_wrapper.h"
#include <algorithm>
#include <map>
#include <string>
#include "lite/backends/opencl/cl_include.h"
#include "lite/backends/opencl/cl_runtime.h"
#include "lite/backends/opencl/cl_utility.h"
namespace paddle {
namespace lite {

static cl_channel_type GetCLChannelType(const PrecisionType type) {
  switch (type) {
    case PRECISION(kFloat):
      return CL_FLOAT;
    case PRECISION(kFP16):
      return CL_HALF_FLOAT;
    case PRECISION(kInt32):
      return CL_SIGNED_INT32;
    case PRECISION(kInt8):
      return CL_SIGNED_INT8;
    default:
      LOG(FATAL) << "Unsupported image channel type: " << PrecisionToStr(type);
      return 0;
  }
}

bool BufferValid(const size_t req_size) {
  bool valid = true;
  std::map<std::string, size_t> &dev_map = CLRuntime::Global()->GetDeviceInfo();
  const size_t max_global_mem_size =
      dev_map["CL_DEVICE_GLOBAL_MEM_SIZE_KB"] * 1000.;
  if (req_size > max_global_mem_size) {
    std::string log = "malloc buffer is out of max global mem size(byte):" +
                      std::to_string(max_global_mem_size) +
                      ", but required global mem size(byte):" +
                      std::to_string(req_size);
    std::cout << log << std::endl;
    LOG(FATAL) << log;
    valid = false;
  }
  return valid;
}

void *TargetWrapperCL::Malloc(size_t size) {
  BufferValid(size);
  cl_int status;
  cl::Buffer *buffer = new cl::Buffer(CLRuntime::Global()->context(),
                                      CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
                                      size,
                                      nullptr,
                                      &status);
  if (status != CL_SUCCESS) {
    delete buffer;
    buffer = nullptr;
  }
  CL_CHECK_FATAL(status);
  return buffer;
}

void TargetWrapperCL::Free(void *ptr) {
  if (ptr != nullptr) {
    cl::Buffer *cl_buffer = static_cast<cl::Buffer *>(ptr);
    delete cl_buffer;
  }
}

bool ImageValid(const size_t req_img_w, const size_t req_img_h) {
  bool valid = true;
  std::map<std::string, size_t> &dev_map = CLRuntime::Global()->GetDeviceInfo();
  auto support_img = dev_map["CL_DEVICE_IMAGE_SUPPORT"];
  if (!support_img) {
    LOG(FATAL) << "device does not support opencl image: " << support_img;
    valid = false;
  }
  auto max_img_w = dev_map["CL_DEVICE_IMAGE2D_MAX_WIDTH"];
  auto max_img_h = dev_map["CL_DEVICE_IMAGE2D_MAX_HEIGHT"];
  if (req_img_w > max_img_w || req_img_h > max_img_h) {
    std::string log = "malloc image is out of max image size(w,h):" +
                      std::to_string(max_img_w) + "," +
                      std::to_string(max_img_h) + ", need image size(w,h):" +
                      std::to_string(req_img_w) + "," +
                      std::to_string(req_img_h);
    std::cout << log << std::endl;
    LOG(FATAL) << log;
    valid = false;
  }
  return valid;
}

template <>
void *TargetWrapperCL::MallocImage<float>(const size_t cl_image2d_width,
                                          const size_t cl_image2d_height,
                                          void *host_ptr) {
  ImageValid(cl_image2d_width, cl_image2d_height);
  cl::ImageFormat img_format(CL_RGBA, GetCLChannelType(PRECISION(kFloat)));
  cl_int status;
  cl::Image2D *cl_image = new cl::Image2D(
      CLRuntime::Global()->context(),
      (host_ptr ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE) |
          (host_ptr ? CL_MEM_COPY_HOST_PTR : CL_MEM_ALLOC_HOST_PTR),
      img_format,
      cl_image2d_width,
      cl_image2d_height,
      0,
      host_ptr,
      &status);
  if (status != CL_SUCCESS) {
    delete cl_image;
    cl_image = nullptr;
  }
  CL_CHECK_FATAL(status);
  return cl_image;
}

template <>  // use uint16_t represents half float
void *TargetWrapperCL::MallocImage<uint16_t>(const size_t cl_image2d_width,
                                             const size_t cl_image2d_height,
                                             void *host_ptr) {
  ImageValid(cl_image2d_width, cl_image2d_height);
  cl::ImageFormat img_format(CL_RGBA, GetCLChannelType(PRECISION(kFP16)));
  cl_int status;
  cl::Image2D *cl_image = new cl::Image2D(
      CLRuntime::Global()->context(),
      (host_ptr ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE) |
          (host_ptr ? CL_MEM_COPY_HOST_PTR : CL_MEM_ALLOC_HOST_PTR),
      img_format,
      cl_image2d_width,
      cl_image2d_height,
      0,
      host_ptr,
      &status);
  if (status != CL_SUCCESS) {
    delete cl_image;
    cl_image = nullptr;
  }
  CL_CHECK_FATAL(status);
  return cl_image;
}

template <>
void *TargetWrapperCL::MallocImage<int32_t>(const size_t cl_image2d_width,
                                            const size_t cl_image2d_height,
                                            void *host_ptr) {
  ImageValid(cl_image2d_width, cl_image2d_height);
  cl::ImageFormat img_format(CL_RGBA, GetCLChannelType(PRECISION(kInt32)));
  cl_int status;
  cl::Image2D *cl_image = new cl::Image2D(
      CLRuntime::Global()->context(),
      (host_ptr ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE) |
          (host_ptr ? CL_MEM_COPY_HOST_PTR : CL_MEM_ALLOC_HOST_PTR),
      img_format,
      cl_image2d_width,
      cl_image2d_height,
      0,
      host_ptr,
      &status);
  if (status != CL_SUCCESS) {
    delete cl_image;
    cl_image = nullptr;
  }
  CL_CHECK_FATAL(status);
  return cl_image;
}

void TargetWrapperCL::FreeImage(void *image) {
  if (image != nullptr) {
    cl::Image2D *cl_image = static_cast<cl::Image2D *>(image);
    delete cl_image;
  }
}

void *TargetWrapperCL::Map(void *buffer, size_t offset, size_t size) {
  BufferValid(size);
  cl::Buffer *cl_buffer = static_cast<cl::Buffer *>(buffer);
  cl_int status;
  void *mapped_ptr = CLRuntime::Global()->command_queue().enqueueMapBuffer(
      *cl_buffer,
      CL_TRUE,
      CL_MAP_READ | CL_MAP_WRITE,
      offset,
      size,
      nullptr,
      nullptr,
      &status);
  if (status != CL_SUCCESS) {
    mapped_ptr = nullptr;
  }
  CL_CHECK_FATAL(status);
  return mapped_ptr;
}

void *TargetWrapperCL::MapImage(void *image,
                                const size_t cl_image2d_width,
                                const size_t cl_image2d_height,
                                size_t cl_image2d_row_pitch,
                                size_t cl_image2d_slice_pitch) {
  ImageValid(cl_image2d_width, cl_image2d_height);
  cl::Image2D *cl_image = static_cast<cl::Image2D *>(image);
  cl::array<size_t, 3> origin = {0, 0, 0};
  cl::array<size_t, 3> region = {cl_image2d_width, cl_image2d_height, 1};
  cl_int status;
  void *mapped_ptr = CLRuntime::Global()->command_queue().enqueueMapImage(
      *cl_image,
      CL_TRUE,
      CL_MAP_READ | CL_MAP_WRITE,
      origin,
      region,
      &cl_image2d_row_pitch,
      &cl_image2d_slice_pitch,
      nullptr,
      nullptr,
      &status);
  if (status != CL_SUCCESS) {
    mapped_ptr = nullptr;
  }
  CL_CHECK_FATAL(status);
  return mapped_ptr;
}

void TargetWrapperCL::Unmap(void *cl_obj, void *mapped_ptr) {
  cl::Memory *mem_obj = static_cast<cl::Memory *>(cl_obj);
  cl_int status = CLRuntime::Global()->command_queue().enqueueUnmapMemObject(
      *mem_obj, mapped_ptr, nullptr, nullptr);
  CL_CHECK_FATAL(status);
}

void TargetWrapperCL::MemcpySync(void *dst,
                                 const void *src,
                                 size_t size,
                                 IoDirection dir) {
  BufferValid(size);
  cl_int status;
  auto stream = CLRuntime::Global()->command_queue();
  switch (dir) {
    case IoDirection::DtoD:
      status = stream.enqueueCopyBuffer(*static_cast<const cl::Buffer *>(src),
                                        *static_cast<cl::Buffer *>(dst),
                                        0,
                                        0,
                                        size,
                                        nullptr,
                                        nullptr);
      CL_CHECK_FATAL(status);
      CLRuntime::Global()->command_queue().finish();
      break;
    case IoDirection::HtoD:
      status = stream.enqueueWriteBuffer(*static_cast<cl::Buffer *>(dst),
                                         CL_TRUE,
                                         0,
                                         size,
                                         src,
                                         nullptr,
                                         nullptr);
      CL_CHECK_FATAL(status);
      break;
    case IoDirection::DtoH:
      status = stream.enqueueReadBuffer(*static_cast<const cl::Buffer *>(src),
                                        CL_TRUE,
                                        0,
                                        size,
                                        dst,
                                        nullptr,
                                        nullptr);
      CL_CHECK_FATAL(status);
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
  }
}

void TargetWrapperCL::MemcpyAsync(void *dst,
                                  const void *src,
                                  size_t size,
                                  IoDirection dir,
                                  const stream_t &stream) {
  BufferValid(size);
  cl_int status;
  switch (dir) {
    case IoDirection::DtoD:
      status = stream.enqueueCopyBuffer(*static_cast<const cl::Buffer *>(src),
                                        *static_cast<cl::Buffer *>(dst),
                                        0,
                                        0,
                                        size,
                                        nullptr,
                                        nullptr);
      CL_CHECK_FATAL(status);
      break;
    case IoDirection::HtoD:
      status = stream.enqueueWriteBuffer(*static_cast<cl::Buffer *>(dst),
                                         CL_FALSE,
                                         0,
                                         size,
                                         src,
                                         nullptr,
                                         nullptr);
      CL_CHECK_FATAL(status);
      break;
    case IoDirection::DtoH:
      status = stream.enqueueReadBuffer(*static_cast<const cl::Buffer *>(src),
                                        CL_FALSE,
                                        0,
                                        size,
                                        dst,
                                        nullptr,
                                        nullptr);
      CL_CHECK_FATAL(status);
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
  }
}

void TargetWrapperCL::ImgcpySync(void *dst,
                                 const void *src,
                                 const size_t cl_image2d_width,
                                 const size_t cl_image2d_height,
                                 const size_t cl_image2d_row_pitch,
                                 const size_t cl_image2d_slice_pitch,
                                 IoDirection dir) {
  ImageValid(cl_image2d_width, cl_image2d_height);
  cl::array<size_t, 3> origin = {0, 0, 0};
  cl::array<size_t, 3> region = {cl_image2d_width, cl_image2d_height, 1};
  cl_int status;
  auto stream = CLRuntime::Global()->command_queue();
  switch (dir) {
    case IoDirection::DtoD:
      status = stream.enqueueCopyImage(*static_cast<const cl::Image2D *>(src),
                                       *static_cast<cl::Image2D *>(dst),
                                       origin,
                                       origin,
                                       region,
                                       nullptr,
                                       nullptr);
      CL_CHECK_FATAL(status);
      CLRuntime::Global()->command_queue().finish();
      break;
    case IoDirection::HtoD:
      status = stream.enqueueWriteImage(*static_cast<cl::Image2D *>(dst),
                                        CL_TRUE,
                                        origin,
                                        region,
                                        cl_image2d_row_pitch,
                                        cl_image2d_slice_pitch,
                                        src,
                                        nullptr,
                                        nullptr);
      CL_CHECK_FATAL(status);
      break;
    case IoDirection::DtoH:
      status = stream.enqueueReadImage(*static_cast<const cl::Image2D *>(src),
                                       CL_TRUE,
                                       origin,
                                       region,
                                       cl_image2d_row_pitch,
                                       cl_image2d_slice_pitch,
                                       dst,
                                       nullptr,
                                       nullptr);
      CL_CHECK_FATAL(status);
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
  }
}

void TargetWrapperCL::ImgcpyAsync(void *dst,
                                  const void *src,
                                  const size_t cl_image2d_width,
                                  const size_t cl_image2d_height,
                                  const size_t cl_image2d_row_pitch,
                                  const size_t cl_image2d_slice_pitch,
                                  IoDirection dir,
                                  const stream_t &stream) {
  ImageValid(cl_image2d_width, cl_image2d_height);
  cl::array<size_t, 3> origin = {0, 0, 0};
  cl::array<size_t, 3> region = {cl_image2d_width, cl_image2d_height, 1};
  cl_int status;
  switch (dir) {
    case IoDirection::DtoD:
      status = stream.enqueueCopyImage(*static_cast<const cl::Image2D *>(src),
                                       *static_cast<cl::Image2D *>(dst),
                                       origin,
                                       origin,
                                       region,
                                       nullptr,
                                       nullptr);
      CL_CHECK_FATAL(status);
      break;
    case IoDirection::HtoD:
      status = stream.enqueueWriteImage(*static_cast<cl::Image2D *>(dst),
                                        CL_FALSE,
                                        origin,
                                        region,
                                        cl_image2d_row_pitch,
                                        cl_image2d_slice_pitch,
                                        src,
                                        nullptr,
                                        nullptr);
      CL_CHECK_FATAL(status);
      break;
    case IoDirection::DtoH:
      status = stream.enqueueReadImage(*static_cast<const cl::Image2D *>(src),
                                       CL_FALSE,
                                       origin,
                                       region,
                                       cl_image2d_row_pitch,
                                       cl_image2d_slice_pitch,
                                       dst,
                                       nullptr,
                                       nullptr);
      CL_CHECK_FATAL(status);
      break;
    default:
      LOG(FATAL) << "Unsupported IoDirection " << static_cast<int>(dir);
  }
}

}  // namespace lite
}  // namespace paddle
