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

#include "lite/backends/opencl/cl_include.h"
#include "lite/utils/log/cp_logging.h"

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 200
#endif
#ifndef CL_HPP_TARGET_OPENCL_VERSION
#define CL_HPP_TARGET_OPENCL_VERSION 110
#endif
#ifndef CL_HPP_MINIMUM_OPENCL_VERSION
#define CL_HPP_MINIMUM_OPENCL_VERSION 110
#endif

#if CL_HPP_TARGET_OPENCL_VERSION < 200
#define CL_API_SUFFIX__VERSION_2_0
#endif

namespace paddle {
namespace lite {

class CLWrapper final {
 public:
  static CLWrapper *Global();
  // Platform APIs
  using clGetPlatformIDsType = cl_int (*)(cl_uint, cl_platform_id *, cl_uint *);
  using clGetPlatformInfoType =
      cl_int (*)(cl_platform_id, cl_platform_info, size_t, void *, size_t *);
  using clBuildProgramType = cl_int (*)(cl_program,
                                        cl_uint,
                                        const cl_device_id *,
                                        const char *,
                                        void(CL_CALLBACK *)(  // NOLINT
                                            cl_program,
                                            void *),
                                        void *);
  using clEnqueueNDRangeKernelType = cl_int (*)(cl_command_queue,
                                                cl_kernel,
                                                cl_uint,
                                                const size_t *,
                                                const size_t *,
                                                const size_t *,
                                                cl_uint,
                                                const cl_event *,
                                                cl_event *);
  using clSetKernelArgType = cl_int (*)(cl_kernel,
                                        cl_uint,
                                        size_t,
                                        const void *);
  using clRetainMemObjectType = cl_int (*)(cl_mem);
  using clReleaseMemObjectType = cl_int (*)(cl_mem);
  using clEnqueueUnmapMemObjectType = cl_int (*)(
      cl_command_queue, cl_mem, void *, cl_uint, const cl_event *, cl_event *);
  using clRetainCommandQueueType = cl_int (*)(cl_command_queue command_queue);
  using clCreateContextType = cl_context (*)(const cl_context_properties *,
                                             cl_uint,
                                             const cl_device_id *,
                                             void(CL_CALLBACK *)(  // NOLINT
                                                 const char *,
                                                 const void *,
                                                 size_t,
                                                 void *),
                                             void *,
                                             cl_int *);
  using clCreateContextFromTypeType =
      cl_context (*)(const cl_context_properties *,
                     cl_device_type,
                     void(CL_CALLBACK *)(  // NOLINT
                         const char *,
                         const void *,
                         size_t,
                         void *),
                     void *,
                     cl_int *);
  using clReleaseContextType = cl_int (*)(cl_context);
  using clWaitForEventsType = cl_int (*)(cl_uint, const cl_event *);
  using clReleaseEventType = cl_int (*)(cl_event);
  using clEnqueueWriteBufferType = cl_int (*)(cl_command_queue,
                                              cl_mem,
                                              cl_bool,
                                              size_t,
                                              size_t,
                                              const void *,
                                              cl_uint,
                                              const cl_event *,
                                              cl_event *);
  using clEnqueueReadBufferType = cl_int (*)(cl_command_queue,
                                             cl_mem,
                                             cl_bool,
                                             size_t,
                                             size_t,
                                             void *,
                                             cl_uint,
                                             const cl_event *,
                                             cl_event *);
  using clEnqueueReadImageType = cl_int (*)(cl_command_queue,
                                            cl_mem,
                                            cl_bool,
                                            const size_t *,
                                            const size_t *,
                                            size_t,
                                            size_t,
                                            void *,
                                            cl_uint,
                                            const cl_event *,
                                            cl_event *);
  using clGetProgramBuildInfoType = cl_int (*)(cl_program,
                                               cl_device_id,
                                               cl_program_build_info,
                                               size_t,
                                               void *,
                                               size_t *);
  using clRetainProgramType = cl_int (*)(cl_program program);
  using clEnqueueMapBufferType = void *(*)(cl_command_queue,
                                           cl_mem,
                                           cl_bool,
                                           cl_map_flags,
                                           size_t,
                                           size_t,
                                           cl_uint,
                                           const cl_event *,
                                           cl_event *,
                                           cl_int *);
  using clEnqueueMapImageType = void *(*)(cl_command_queue,
                                          cl_mem,
                                          cl_bool,
                                          cl_map_flags,
                                          const size_t *,
                                          const size_t *,
                                          size_t *,
                                          size_t *,
                                          cl_uint,
                                          const cl_event *,
                                          cl_event *,
                                          cl_int *);
  using clCreateCommandQueueType = cl_command_queue(CL_API_CALL *)(  // NOLINT
      cl_context,
      cl_device_id,
      cl_command_queue_properties,
      cl_int *);
  using clGetCommandQueueInfoType = cl_int (*)(
      cl_command_queue, cl_command_queue_info, size_t, void *, size_t *);
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
  using clCreateCommandQueueWithPropertiesType = cl_command_queue (*)(
      cl_context, cl_device_id, const cl_queue_properties *, cl_int *);
#endif
  using clReleaseCommandQueueType = cl_int (*)(cl_command_queue);
  using clCreateProgramWithBinaryType = cl_program (*)(cl_context,
                                                       cl_uint,
                                                       const cl_device_id *,
                                                       const size_t *,
                                                       const unsigned char **,
                                                       cl_int *,
                                                       cl_int *);
  using clRetainContextType = cl_int (*)(cl_context context);
  using clGetContextInfoType =
      cl_int (*)(cl_context, cl_context_info, size_t, void *, size_t *);
  using clReleaseProgramType = cl_int (*)(cl_program program);
  using clFlushType = cl_int (*)(cl_command_queue command_queue);
  using clFinishType = cl_int (*)(cl_command_queue command_queue);
  using clGetProgramInfoType =
      cl_int (*)(cl_program, cl_program_info, size_t, void *, size_t *);
  using clCreateKernelType = cl_kernel (*)(cl_program, const char *, cl_int *);
  using clRetainKernelType = cl_int (*)(cl_kernel kernel);
  using clCreateBufferType =
      cl_mem (*)(cl_context, cl_mem_flags, size_t, void *, cl_int *);
  using clCreateImage2DType = cl_mem(CL_API_CALL *)(cl_context,  // NOLINT
                                                    cl_mem_flags,
                                                    const cl_image_format *,
                                                    size_t,
                                                    size_t,
                                                    size_t,
                                                    void *,
                                                    cl_int *);
  using clCreateImageType = cl_mem (*)(cl_context,
                                       cl_mem_flags,
                                       const cl_image_format *,
                                       const cl_image_desc *,
                                       void *,
                                       cl_int *);
  using clCreateUserEventType = cl_event (*)(cl_context, cl_int *);
  using clCreateProgramWithSourceType = cl_program (*)(
      cl_context, cl_uint, const char **, const size_t *, cl_int *);
  using clReleaseKernelType = cl_int (*)(cl_kernel kernel);
  using clGetDeviceInfoType =
      cl_int (*)(cl_device_id, cl_device_info, size_t, void *, size_t *);
  using clGetDeviceIDsType = cl_int (*)(
      cl_platform_id, cl_device_type, cl_uint, cl_device_id *, cl_uint *);
  using clRetainDeviceType = cl_int (*)(cl_device_id);
  using clReleaseDeviceType = cl_int (*)(cl_device_id);
  using clRetainEventType = cl_int (*)(cl_event);
  using clGetKernelWorkGroupInfoType = cl_int (*)(cl_kernel,
                                                  cl_device_id,
                                                  cl_kernel_work_group_info,
                                                  size_t,
                                                  void *,
                                                  size_t *);
  using clGetEventInfoType = cl_int (*)(cl_event event,
                                        cl_event_info param_name,
                                        size_t param_value_size,
                                        void *param_value,
                                        size_t *param_value_size_ret);
  using clGetEventProfilingInfoType = cl_int (*)(cl_event event,
                                                 cl_profiling_info param_name,
                                                 size_t param_value_size,
                                                 void *param_value,
                                                 size_t *param_value_size_ret);
  using clGetImageInfoType =
      cl_int (*)(cl_mem, cl_image_info, size_t, void *, size_t *);

  using clGetMemObjectInfoType =
      cl_int (*)(cl_mem, cl_mem_info, size_t, void *, size_t *);

  using clEnqueueCopyBufferType = cl_int (*)(cl_command_queue,
                                             cl_mem,
                                             cl_mem,
                                             size_t,
                                             size_t,
                                             size_t,
                                             cl_uint,
                                             const cl_event *,
                                             cl_event *);
  using clEnqueueWriteImageType = cl_int (*)(cl_command_queue,
                                             cl_mem,
                                             cl_bool,
                                             const size_t *,
                                             const size_t *,
                                             size_t,
                                             size_t,
                                             const void *,
                                             cl_uint,
                                             const cl_event *,
                                             cl_event *);
  using clEnqueueCopyImageType = cl_int (*)(cl_command_queue,
                                            cl_mem,
                                            cl_mem,
                                            const size_t *,
                                            const size_t *,
                                            const size_t *,
                                            cl_uint,
                                            const cl_event *,
                                            cl_event *);

  clGetPlatformIDsType clGetPlatformIDs() {
    if (clGetPlatformIDs_ == nullptr) {
      LOG(ERROR) << "Cannot load clGetPlatformIDs!";
    }
    return clGetPlatformIDs_;
  }

  clGetPlatformInfoType clGetPlatformInfo() {
    if (clGetPlatformInfo_ == nullptr) {
      LOG(ERROR) << "Cannot load clGetPlatformInfo!";
    }
    return clGetPlatformInfo_;
  }

  clBuildProgramType clBuildProgram() {
    if (clBuildProgram_ == nullptr) {
      LOG(ERROR) << "Cannot load clBuildProgram!";
    }
    return clBuildProgram_;
  }

  clEnqueueNDRangeKernelType clEnqueueNDRangeKernel() {
    if (clEnqueueNDRangeKernel_ == nullptr) {
      LOG(ERROR) << "Cannot load clEnqueueNDRangeKernel!";
    }
    return clEnqueueNDRangeKernel_;
  }

  clSetKernelArgType clSetKernelArg() {
    if (clSetKernelArg_ == nullptr) {
      LOG(ERROR) << "Cannot load clSetKernelArg!";
    }
    return clSetKernelArg_;
  }

  clRetainMemObjectType clRetainMemObject() {
    if (clRetainMemObject_ == nullptr) {
      LOG(ERROR) << "Cannot load clRetainMemObject!";
    }
    return clRetainMemObject_;
  }

  clReleaseMemObjectType clReleaseMemObject() {
    if (clReleaseMemObject_ == nullptr) {
      LOG(ERROR) << "Cannot load clReleaseMemObject!";
    }
    return clReleaseMemObject_;
  }

  clEnqueueUnmapMemObjectType clEnqueueUnmapMemObject() {
    if (clEnqueueUnmapMemObject_ == nullptr) {
      LOG(ERROR) << "Cannot load clEnqueueUnmapMemObject!";
    }
    return clEnqueueUnmapMemObject_;
  }

  clRetainCommandQueueType clRetainCommandQueue() {
    if (clRetainCommandQueue_ == nullptr) {
      LOG(ERROR) << "Cannot load clRetainCommandQueue!";
    }
    return clRetainCommandQueue_;
  }

  clCreateContextType clCreateContext() {
    if (clCreateContext_ == nullptr) {
      LOG(ERROR) << "Cannot load clCreateContext!";
    }
    return clCreateContext_;
  }

  clCreateContextFromTypeType clCreateContextFromType() {
    if (clCreateContextFromType_ == nullptr) {
      LOG(ERROR) << "Cannot load clCreateContextFromType!";
    }
    return clCreateContextFromType_;
  }

  clReleaseContextType clReleaseContext() {
    if (clReleaseContext_ == nullptr) {
      LOG(ERROR) << "Cannot load clReleaseContext!";
    }
    return clReleaseContext_;
  }

  clWaitForEventsType clWaitForEvents() {
    if (clWaitForEvents_ == nullptr) {
      LOG(ERROR) << "Cannot load clWaitForEvents!";
    }
    return clWaitForEvents_;
  }

  clReleaseEventType clReleaseEvent() {
    if (clReleaseEvent_ == nullptr) {
      LOG(ERROR) << "Cannot load clReleaseEvent!";
    }
    return clReleaseEvent_;
  }

  clEnqueueWriteBufferType clEnqueueWriteBuffer() {
    if (clEnqueueWriteBuffer_ == nullptr) {
      LOG(ERROR) << "Cannot loadcl clEnqueueWriteBuffer!";
    }
    return clEnqueueWriteBuffer_;
  }

  clEnqueueReadBufferType clEnqueueReadBuffer() {
    if (clEnqueueReadBuffer_ == nullptr) {
      LOG(ERROR) << "Cannot load clEnqueueReadBuffer!";
    }
    return clEnqueueReadBuffer_;
  }

  clEnqueueReadImageType clEnqueueReadImage() {
    if (clEnqueueReadImage_ == nullptr) {
      LOG(ERROR) << "Cannot load clEnqueueReadImage!";
    }
    return clEnqueueReadImage_;
  }

  clGetProgramBuildInfoType clGetProgramBuildInfo() {
    if (clGetProgramBuildInfo_ == nullptr) {
      LOG(ERROR) << "Cannot load clGetProgramBuildInfo!";
    }
    return clGetProgramBuildInfo_;
  }

  clRetainProgramType clRetainProgram() {
    if (clRetainProgram_ == nullptr) {
      LOG(ERROR) << "Cannot load clRetainProgram!";
    }
    return clRetainProgram_;
  }

  clEnqueueMapBufferType clEnqueueMapBuffer() {
    if (clEnqueueMapBuffer_ == nullptr) {
      LOG(ERROR) << "Cannot load clEnqueueMapBuffer!";
    }
    return clEnqueueMapBuffer_;
  }

  clEnqueueMapImageType clEnqueueMapImage() {
    if (clEnqueueMapImage_ == nullptr) {
      LOG(ERROR) << "Cannot load clEnqueueMapImage!";
    }
    return clEnqueueMapImage_;
  }

  clCreateCommandQueueType clCreateCommandQueue() {
    if (clCreateCommandQueue_ == nullptr) {
      LOG(ERROR) << "Cannot load clCreateCommandQueue!";
    }
    return clCreateCommandQueue_;
  }

  clGetCommandQueueInfoType clGetCommandQueueInfo() {
    if (clGetCommandQueueInfo_ == nullptr) {
      LOG(ERROR) << "Cannot load clGetCommandQueueInfo!";
    }
    return clGetCommandQueueInfo_;
  }

#if CL_HPP_TARGET_OPENCL_VERSION >= 200

  clCreateCommandQueueWithPropertiesType clCreateCommandQueueWithProperties() {
    if (clCreateCommandQueueWithProperties_ == nullptr) {
      LOG(ERROR) << "Cannot load clCreateCommandQueueWithProperties!";
    }
    return clCreateCommandQueueWithProperties_;
  }

#endif

  clReleaseCommandQueueType clReleaseCommandQueue() {
    if (clReleaseCommandQueue_ == nullptr) {
      LOG(ERROR) << "Cannot load clReleaseCommandQueue!";
    }
    return clReleaseCommandQueue_;
  }

  clCreateProgramWithBinaryType clCreateProgramWithBinary() {
    if (clCreateProgramWithBinary_ == nullptr) {
      LOG(ERROR) << "Cannot load clCreateProgramWithBinary!";
    }
    return clCreateProgramWithBinary_;
  }

  clRetainContextType clRetainContext() {
    if (clRetainContext_ == nullptr) {
      LOG(ERROR) << "Cannot load clRetainContext!";
    }
    return clRetainContext_;
  }

  clGetContextInfoType clGetContextInfo() {
    if (clGetContextInfo_ == nullptr) {
      LOG(ERROR) << "Cannot load clGetContextInfo!";
    }
    return clGetContextInfo_;
  }

  clReleaseProgramType clReleaseProgram() {
    if (clReleaseProgram_ == nullptr) {
      LOG(ERROR) << "Cannot load clReleaseProgram!";
    }
    return clReleaseProgram_;
  }

  clFlushType clFlush() {
    if (clFlush_ == nullptr) {
      LOG(ERROR) << "Cannot load clFlush!";
    }
    return clFlush_;
  }

  clFinishType clFinish() {
    if (clFinish_ == nullptr) {
      LOG(ERROR) << "Cannot load clFinish!";
    }
    return clFinish_;
  }

  clGetProgramInfoType clGetProgramInfo() {
    if (clGetProgramInfo_ == nullptr) {
      LOG(ERROR) << "Cannot load clGetProgramInfo!";
    }
    return clGetProgramInfo_;
  }

  clCreateKernelType clCreateKernel() {
    if (clCreateKernel_ == nullptr) {
      LOG(ERROR) << "Cannot load clCreateKernel!";
    }
    return clCreateKernel_;
  }

  clRetainKernelType clRetainKernel() {
    if (clRetainKernel_ == nullptr) {
      LOG(ERROR) << "Cannot load clRetainKernel!";
    }
    return clRetainKernel_;
  }

  clCreateBufferType clCreateBuffer() {
    if (clCreateBuffer_ == nullptr) {
      LOG(ERROR) << "Cannot load clCreateBuffer!";
    }
    return clCreateBuffer_;
  }

  clCreateImage2DType clCreateImage2D() {
    if (clCreateImage2D_ == nullptr) {
      LOG(ERROR) << "Cannot load clCreateImage2D!";
    }
    return clCreateImage2D_;
  }

#if CL_HPP_TARGET_OPENCL_VERSION >= 120

  clCreateImageType clCreateImage() {
    if (clCreateImage_ == nullptr) {
      LOG(ERROR) << "Cannot load clCreateImage!";
    }
    return clCreateImage_;
  }

#endif

  clCreateUserEventType clCreateUserEvent() {
    if (clCreateUserEvent_ == nullptr) {
      LOG(ERROR) << "Cannot load clCreateUserEvent!";
    }
    return clCreateUserEvent_;
  }

  clCreateProgramWithSourceType clCreateProgramWithSource() {
    if (clCreateProgramWithSource_ == nullptr) {
      LOG(ERROR) << "Cannot load clCreateProgramWithSource!";
    }
    return clCreateProgramWithSource_;
  }

  clReleaseKernelType clReleaseKernel() {
    if (clReleaseKernel_ == nullptr) {
      LOG(ERROR) << "Cannot load clReleaseKernel!";
    }
    return clReleaseKernel_;
  }

  clGetDeviceInfoType clGetDeviceInfo() {
    if (clGetDeviceInfo_ == nullptr) {
      LOG(ERROR) << "Cannot load clGetDeviceInfo!";
    }
    return clGetDeviceInfo_;
  }

  clGetDeviceIDsType clGetDeviceIDs() {
    if (clGetDeviceIDs_ == nullptr) {
      LOG(ERROR) << "Cannot load clGetDeviceIDs!";
    }
    return clGetDeviceIDs_;
  }

#if CL_HPP_TARGET_OPENCL_VERSION >= 120

  clRetainDeviceType clRetainDevice() {
    if (clRetainDevice_ == nullptr) {
      LOG(ERROR) << "Cannot load clRetainDevice!";
    }
    return clRetainDevice_;
  }

  clReleaseDeviceType clReleaseDevice() {
    if (clReleaseDevice_ == nullptr) {
      LOG(ERROR) << "Cannot load clReleaseDevice!";
    }
    return clReleaseDevice_;
  }

#endif

  clRetainEventType clRetainEvent() {
    if (clRetainEvent_ == nullptr) {
      LOG(ERROR) << "Cannot load clRetainEvent!";
    }
    return clRetainEvent_;
  }

  clGetKernelWorkGroupInfoType clGetKernelWorkGroupInfo() {
    if (clGetKernelWorkGroupInfo_ == nullptr) {
      LOG(ERROR) << "Cannot load clGetKernelWorkGroupInfo!";
    }
    return clGetKernelWorkGroupInfo_;
  }

  clGetEventInfoType clGetEventInfo() {
    if (clGetEventInfo_ == nullptr) {
      LOG(ERROR) << "Cannot load clGetEventInfo!";
    }
    return clGetEventInfo_;
  }

  clGetEventProfilingInfoType clGetEventProfilingInfo() {
    if (clGetEventProfilingInfo_ == nullptr) {
      LOG(ERROR) << "Cannot load clGetEventProfilingInfo!";
    }
    return clGetEventProfilingInfo_;
  }

  clGetImageInfoType clGetImageInfo() {
    if (clGetImageInfo_ == nullptr) {
      LOG(ERROR) << "Cannot load clGetImageInfo!";
    }
    return clGetImageInfo_;
  }

  clGetMemObjectInfoType clGetMemObjectInfo() {
    if (clGetMemObjectInfo_ == nullptr) {
      LOG(ERROR) << "Cannot load clGetMemObjectInfo!";
    }
    return clGetMemObjectInfo_;
  }

  clEnqueueCopyBufferType clEnqueueCopyBuffer() {
    if (clEnqueueCopyBuffer_ == nullptr) {
      LOG(ERROR) << "Cannot load clEnqueueCopyBuffer!";
    }
    return clEnqueueCopyBuffer_;
  }

  clEnqueueWriteImageType clEnqueueWriteImage() {
    if (clEnqueueWriteImage_ == nullptr) {
      LOG(ERROR) << "Cannot load clEnqueueWriteImage!";
    }
    return clEnqueueWriteImage_;
  }

  clEnqueueCopyImageType clEnqueueCopyImage() {
    if (clEnqueueCopyImage_ == nullptr) {
      LOG(ERROR) << "Cannot load clEnqueueCopyImage!";
    }
    return clEnqueueCopyImage_;
  }

  bool OpenclLibFound() { return opencl_lib_found_; }

  bool DlsymSuccess() { return dlsym_success_; }

 private:
  CLWrapper();
  CLWrapper(const CLWrapper &) = delete;
  CLWrapper &operator=(const CLWrapper &) = delete;
  bool InitHandle();
  bool InitFunctions();
  bool opencl_lib_found_{true};
  bool is_first_init_{true};
  bool dlsym_success_{true};
  void *handle_{nullptr};

  clGetPlatformIDsType clGetPlatformIDs_{nullptr};
  clGetPlatformInfoType clGetPlatformInfo_{nullptr};
  clBuildProgramType clBuildProgram_{nullptr};
  clEnqueueNDRangeKernelType clEnqueueNDRangeKernel_{nullptr};
  clSetKernelArgType clSetKernelArg_{nullptr};
  clRetainMemObjectType clRetainMemObject_{nullptr};
  clReleaseMemObjectType clReleaseMemObject_{nullptr};
  clEnqueueUnmapMemObjectType clEnqueueUnmapMemObject_{nullptr};
  clRetainCommandQueueType clRetainCommandQueue_{nullptr};
  clCreateContextType clCreateContext_{nullptr};
  clCreateContextFromTypeType clCreateContextFromType_{nullptr};
  clReleaseContextType clReleaseContext_{nullptr};
  clWaitForEventsType clWaitForEvents_{nullptr};
  clReleaseEventType clReleaseEvent_{nullptr};
  clEnqueueWriteBufferType clEnqueueWriteBuffer_{nullptr};
  clEnqueueReadBufferType clEnqueueReadBuffer_{nullptr};
  clEnqueueReadImageType clEnqueueReadImage_{nullptr};
  clGetProgramBuildInfoType clGetProgramBuildInfo_{nullptr};
  clRetainProgramType clRetainProgram_{nullptr};
  clEnqueueMapBufferType clEnqueueMapBuffer_{nullptr};
  clEnqueueMapImageType clEnqueueMapImage_{nullptr};
  clCreateCommandQueueType clCreateCommandQueue_{nullptr};
  clGetCommandQueueInfoType clGetCommandQueueInfo_{nullptr};
#if CL_HPP_TARGET_OPENCL_VERSION >= 200
  clCreateCommandQueueWithPropertiesType clCreateCommandQueueWithProperties_{
      nullptr};
#endif
  clReleaseCommandQueueType clReleaseCommandQueue_{nullptr};
  clCreateProgramWithBinaryType clCreateProgramWithBinary_{nullptr};
  clRetainContextType clRetainContext_{nullptr};
  clGetContextInfoType clGetContextInfo_{nullptr};
  clReleaseProgramType clReleaseProgram_{nullptr};
  clFlushType clFlush_{nullptr};
  clFinishType clFinish_{nullptr};
  clGetProgramInfoType clGetProgramInfo_{nullptr};
  clCreateKernelType clCreateKernel_{nullptr};
  clRetainKernelType clRetainKernel_{nullptr};
  clCreateBufferType clCreateBuffer_{nullptr};
  clCreateImage2DType clCreateImage2D_{nullptr};
  clCreateUserEventType clCreateUserEvent_{nullptr};
  clCreateProgramWithSourceType clCreateProgramWithSource_{nullptr};
  clReleaseKernelType clReleaseKernel_{nullptr};
  clGetDeviceInfoType clGetDeviceInfo_{nullptr};
  clGetDeviceIDsType clGetDeviceIDs_{nullptr};
#if CL_HPP_TARGET_OPENCL_VERSION >= 120
  clRetainDeviceType clRetainDevice_{nullptr};
  clReleaseDeviceType clReleaseDevice_{nullptr};
  clCreateImageType clCreateImage_{nullptr};
#endif
  clRetainEventType clRetainEvent_{nullptr};
  clGetKernelWorkGroupInfoType clGetKernelWorkGroupInfo_{nullptr};
  clGetEventInfoType clGetEventInfo_{nullptr};
  clGetEventProfilingInfoType clGetEventProfilingInfo_{nullptr};
  clGetImageInfoType clGetImageInfo_{nullptr};
  clGetMemObjectInfoType clGetMemObjectInfo_{nullptr};
  clEnqueueCopyBufferType clEnqueueCopyBuffer_{nullptr};
  clEnqueueWriteImageType clEnqueueWriteImage_{nullptr};
  clEnqueueCopyImageType clEnqueueCopyImage_{nullptr};
};
}  // namespace lite
}  // namespace paddle
