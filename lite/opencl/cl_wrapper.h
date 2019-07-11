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

#include "lite/opencl/cl_include.h"

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
                                        void (*pfn_notify)(cl_program, void *),
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
  using clCreateCommandQueueWithPropertiesType = cl_command_queue (*)(
      cl_context, cl_device_id, const cl_queue_properties *, cl_int *);
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

  clGetPlatformIDsType clGetPlatformIDs() { return clGetPlatformIDs_; }

  clGetPlatformInfoType clGetPlatformInfo() { return clGetPlatformInfo_; }

  clBuildProgramType clBuildProgram() { return clBuildProgram_; }

  clEnqueueNDRangeKernelType clEnqueueNDRangeKernel() {
    return clEnqueueNDRangeKernel_;
  }

  clSetKernelArgType clSetKernelArg() { return clSetKernelArg_; }

  clRetainMemObjectType clRetainMemObject() { return clRetainMemObject_; }

  clReleaseMemObjectType clReleaseMemObject() { return clReleaseMemObject_; }

  clEnqueueUnmapMemObjectType clEnqueueUnmapMemObject() {
    return clEnqueueUnmapMemObject_;
  }

  clRetainCommandQueueType clRetainCommandQueue() {
    return clRetainCommandQueue_;
  }

  clCreateContextType clCreateContext() { return clCreateContext_; }

  clCreateContextFromTypeType clCreateContextFromType() {
    return clCreateContextFromType_;
  }

  clReleaseContextType clReleaseContext() { return clReleaseContext_; }

  clWaitForEventsType clWaitForEvents() { return clWaitForEvents_; }

  clReleaseEventType clReleaseEvent() { return clReleaseEvent_; }

  clEnqueueWriteBufferType clEnqueueWriteBuffer() {
    return clEnqueueWriteBuffer_;
  }

  clEnqueueReadBufferType clEnqueueReadBuffer() { return clEnqueueReadBuffer_; }

  clEnqueueReadImageType clEnqueueReadImage() { return clEnqueueReadImage_; }

  clGetProgramBuildInfoType clGetProgramBuildInfo() {
    return clGetProgramBuildInfo_;
  }

  clRetainProgramType clRetainProgram() { return clRetainProgram_; }

  clEnqueueMapBufferType clEnqueueMapBuffer() { return clEnqueueMapBuffer_; }

  clEnqueueMapImageType clEnqueueMapImage() { return clEnqueueMapImage_; }

  clCreateCommandQueueType clCreateCommandQueue() {
    return clCreateCommandQueue_;
  }

  clCreateCommandQueueWithPropertiesType clCreateCommandQueueWithProperties() {
    return clCreateCommandQueueWithProperties_;
  }

  clReleaseCommandQueueType clReleaseCommandQueue() {
    return clReleaseCommandQueue_;
  }

  clCreateProgramWithBinaryType clCreateProgramWithBinary() {
    return clCreateProgramWithBinary_;
  }

  clRetainContextType clRetainContext() { return clRetainContext_; }

  clGetContextInfoType clGetContextInfo() { return clGetContextInfo_; }

  clReleaseProgramType clReleaseProgram() { return clReleaseProgram_; }

  clFlushType clFlush() { return clFlush_; }

  clFinishType clFinish() { return clFinish_; }

  clGetProgramInfoType clGetProgramInfo() { return clGetProgramInfo_; }

  clCreateKernelType clCreateKernel() { return clCreateKernel_; }

  clRetainKernelType clRetainKernel() { return clRetainKernel_; }

  clCreateBufferType clCreateBuffer() { return clCreateBuffer_; }

  clCreateImage2DType clCreateImage2D() { return clCreateImage2D_; }

  clCreateImageType clCreateImage() { return clCreateImage_; }

  clCreateUserEventType clCreateUserEvent() { return clCreateUserEvent_; }

  clCreateProgramWithSourceType clCreateProgramWithSource() {
    return clCreateProgramWithSource_;
  }

  clReleaseKernelType clReleaseKernel() { return clReleaseKernel_; }

  clGetDeviceInfoType clGetDeviceInfo() { return clGetDeviceInfo_; }

  clGetDeviceIDsType clGetDeviceIDs() { return clGetDeviceIDs_; }

  clRetainDeviceType clRetainDevice() { return clRetainDevice_; }

  clReleaseDeviceType clReleaseDevice() { return clReleaseDevice_; }

  clRetainEventType clRetainEvent() { return clRetainEvent_; }

  clGetKernelWorkGroupInfoType clGetKernelWorkGroupInfo() {
    return clGetKernelWorkGroupInfo_;
  }

  clGetEventInfoType clGetEventInfo() { return clGetEventInfo_; }

  clGetEventProfilingInfoType clGetEventProfilingInfo() {
    return clGetEventProfilingInfo_;
  }

  clGetImageInfoType clGetImageInfo() { return clGetImageInfo_; }

  clEnqueueCopyBufferType clEnqueueCopyBuffer() { return clEnqueueCopyBuffer_; }

  clEnqueueWriteImageType clEnqueueWriteImage() { return clEnqueueWriteImage_; }

  clEnqueueCopyImageType clEnqueueCopyImage() { return clEnqueueCopyImage_; }

 private:
  CLWrapper();
  CLWrapper(const CLWrapper &) = delete;
  CLWrapper &operator=(const CLWrapper &) = delete;
  bool InitHandle();
  void InitFunctions();
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
  clCreateCommandQueueWithPropertiesType clCreateCommandQueueWithProperties_{
      nullptr};
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
  clCreateImageType clCreateImage_{nullptr};
  clCreateUserEventType clCreateUserEvent_{nullptr};
  clCreateProgramWithSourceType clCreateProgramWithSource_{nullptr};
  clReleaseKernelType clReleaseKernel_{nullptr};
  clGetDeviceInfoType clGetDeviceInfo_{nullptr};
  clGetDeviceIDsType clGetDeviceIDs_{nullptr};
  clRetainDeviceType clRetainDevice_{nullptr};
  clReleaseDeviceType clReleaseDevice_{nullptr};
  clRetainEventType clRetainEvent_{nullptr};
  clGetKernelWorkGroupInfoType clGetKernelWorkGroupInfo_{nullptr};
  clGetEventInfoType clGetEventInfo_{nullptr};
  clGetEventProfilingInfoType clGetEventProfilingInfo_{nullptr};
  clGetImageInfoType clGetImageInfo_{nullptr};
  clEnqueueCopyBufferType clEnqueueCopyBuffer_{nullptr};
  clEnqueueWriteImageType clEnqueueWriteImage_{nullptr};
  clEnqueueCopyImageType clEnqueueCopyImage_{nullptr};
};
}  // namespace lite
}  // namespace paddle
