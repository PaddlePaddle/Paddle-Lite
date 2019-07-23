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

#include "lite/opencl/cl_wrapper.h"
#include <dlfcn.h>
#include <string>
#include <vector>

namespace paddle {
namespace lite {
CLWrapper *CLWrapper::Global() {
  static CLWrapper wrapper;
  return &wrapper;
}

CLWrapper::CLWrapper() {
  CHECK(InitHandle()) << "Fail to initialize the OpenCL library!";
  InitFunctions();
}

bool CLWrapper::InitHandle() {
  const std::vector<std::string> paths = {
    "libOpenCL.so",
#if defined(__aarch64__)
    // Qualcomm Adreno with Android
    "/system/vendor/lib64/libOpenCL.so",
    "/system/lib64/libOpenCL.so",
    // Arm Mali with Android
    "/system/vendor/lib64/egl/libGLES_mali.so",
    "/system/lib64/egl/libGLES_mali.so",
    // Arm Linux
    "/usr/lib/aarch64-linux-gnu/libOpenCL.so",
#else
    // Qualcomm Adreno with Android
    "/system/vendor/lib/libOpenCL.so",
    "/system/lib/libOpenCL.so",
    // Arm Mali with Android
    "/system/vendor/lib/egl/libGLES_mali.so",
    "/system/lib/egl/libGLES_mali.so",
    // Arm Linux
    "/usr/lib/arm-linux-gnueabihf/libOpenCL.so",
#endif
  };
  std::string target_lib = "Unknown";
  for (auto path : paths) {
    handle_ = dlopen(path.c_str(), RTLD_LAZY);
    if (handle_ != nullptr) {
      target_lib = path;
      break;
    }
  }
  VLOG(4) << "Load the OpenCL library from " << target_lib;
  if (handle_ != nullptr) {
    return true;
  } else {
    return false;
  }
}

void CLWrapper::InitFunctions() {
  CHECK(handle_ != nullptr) << "The library handle can't be null!";
  clGetPlatformIDs_ = (clGetPlatformIDsType)dlsym(handle_, "clGetPlatformIDs");

  clGetPlatformInfo_ =
      (clGetPlatformInfoType)dlsym(handle_, "clGetPlatformInfo");

  clBuildProgram_ = (clBuildProgramType)dlsym(handle_, "clBuildProgram");

  clEnqueueNDRangeKernel_ =
      (clEnqueueNDRangeKernelType)dlsym(handle_, "clEnqueueNDRangeKernel");

  clSetKernelArg_ = (clSetKernelArgType)dlsym(handle_, "clSetKernelArg");

  clRetainMemObject_ =
      (clRetainMemObjectType)dlsym(handle_, "clRetainMemObject");

  clReleaseMemObject_ =
      (clReleaseMemObjectType)dlsym(handle_, "clReleaseMemObject");

  clEnqueueUnmapMemObject_ =
      (clEnqueueUnmapMemObjectType)dlsym(handle_, "clEnqueueUnmapMemObject");

  clRetainCommandQueue_ =
      (clRetainCommandQueueType)dlsym(handle_, "clRetainCommandQueue");

  clCreateContext_ = (clCreateContextType)dlsym(handle_, "clCreateContext");

  clCreateContextFromType_ =
      (clCreateContextFromTypeType)dlsym(handle_, "clCreateContextFromType");

  clReleaseContext_ = (clReleaseContextType)dlsym(handle_, "clReleaseContext");

  clWaitForEvents_ = (clWaitForEventsType)dlsym(handle_, "clWaitForEvents");

  clReleaseEvent_ = (clReleaseEventType)dlsym(handle_, "clReleaseEvent");

  clEnqueueWriteBuffer_ =
      (clEnqueueWriteBufferType)dlsym(handle_, "clEnqueueWriteBuffer");

  clEnqueueReadBuffer_ =
      (clEnqueueReadBufferType)dlsym(handle_, "clEnqueueReadBuffer");

  clEnqueueReadImage_ =
      (clEnqueueReadImageType)dlsym(handle_, "clEnqueueReadImage");

  clGetProgramBuildInfo_ =
      (clGetProgramBuildInfoType)dlsym(handle_, "clGetProgramBuildInfo");

  clRetainProgram_ = (clRetainProgramType)dlsym(handle_, "clRetainProgram");

  clEnqueueMapBuffer_ =
      (clEnqueueMapBufferType)dlsym(handle_, "clEnqueueMapBuffer");

  clEnqueueMapImage_ =
      (clEnqueueMapImageType)dlsym(handle_, "clEnqueueMapImage");

  clCreateCommandQueue_ =
      (clCreateCommandQueueType)dlsym(handle_, "clCreateCommandQueue");

  clCreateCommandQueueWithProperties_ =
      (clCreateCommandQueueWithPropertiesType)dlsym(
          handle_, "clCreateCommandQueueWithProperties");

  clReleaseCommandQueue_ =
      (clReleaseCommandQueueType)dlsym(handle_, "clReleaseCommandQueue");

  clCreateProgramWithBinary_ = (clCreateProgramWithBinaryType)dlsym(
      handle_, "clCreateProgramWithBinary");

  clRetainContext_ = (clRetainContextType)dlsym(handle_, "clRetainContext");

  clGetContextInfo_ = (clGetContextInfoType)dlsym(handle_, "clGetContextInfo");

  clReleaseProgram_ = (clReleaseProgramType)dlsym(handle_, "clReleaseProgram");

  clFlush_ = (clFlushType)dlsym(handle_, "clFlush");

  clFinish_ = (clFinishType)dlsym(handle_, "clFinish");

  clGetProgramInfo_ = (clGetProgramInfoType)dlsym(handle_, "clGetProgramInfo");

  clCreateKernel_ = (clCreateKernelType)dlsym(handle_, "clCreateKernel");

  clRetainKernel_ = (clRetainKernelType)dlsym(handle_, "clRetainKernel");

  clCreateBuffer_ = (clCreateBufferType)dlsym(handle_, "clCreateBuffer");

  clCreateImage2D_ = (clCreateImage2DType)dlsym(handle_, "clCreateImage2D");

  clCreateImage_ = (clCreateImageType)dlsym(handle_, "clCreateImage");

  clCreateUserEvent_ =
      (clCreateUserEventType)dlsym(handle_, "clCreateUserEvent");

  clCreateProgramWithSource_ = (clCreateProgramWithSourceType)dlsym(
      handle_, "clCreateProgramWithSource");

  clReleaseKernel_ = (clReleaseKernelType)dlsym(handle_, "clReleaseKernel");

  clGetDeviceInfo_ = (clGetDeviceInfoType)dlsym(handle_, "clGetDeviceInfo");

  clGetDeviceIDs_ = (clGetDeviceIDsType)dlsym(handle_, "clGetDeviceIDs");

  clRetainDevice_ = (clRetainDeviceType)dlsym(handle_, "clRetainDevice");

  clReleaseDevice_ = (clReleaseDeviceType)dlsym(handle_, "clReleaseDevice");

  clRetainEvent_ = (clRetainEventType)dlsym(handle_, "clRetainEvent");

  clGetKernelWorkGroupInfo_ =
      (clGetKernelWorkGroupInfoType)dlsym(handle_, "clGetKernelWorkGroupInfo");

  clGetEventInfo_ = (clGetEventInfoType)dlsym(handle_, "clGetEventInfo");

  clGetEventProfilingInfo_ =
      (clGetEventProfilingInfoType)dlsym(handle_, "clGetEventProfilingInfo");

  clGetImageInfo_ = (clGetImageInfoType)dlsym(handle_, "clGetImageInfo");

  clEnqueueCopyBuffer_ =
      (clEnqueueCopyBufferType)dlsym(handle_, "clEnqueueCopyBuffer");

  clEnqueueWriteImage_ =
      (clEnqueueWriteImageType)dlsym(handle_, "clEnqueueWriteImage");

  clEnqueueCopyImage_ =
      (clEnqueueCopyImageType)dlsym(handle_, "clEnqueueCopyImage");
}

}  // namespace lite
}  // namespace paddle

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformIDs(cl_uint num_entries,
                                                 cl_platform_id *platforms,
                                                 cl_uint *num_platforms)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clGetPlatformIDs()(
      num_entries, platforms, num_platforms);
}

CL_API_ENTRY cl_int CL_API_CALL clGetPlatformInfo(cl_platform_id platform,
                                                  cl_platform_info param_name,
                                                  size_t param_value_size,
                                                  void *param_value,
                                                  size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clGetPlatformInfo()(
      platform,
      param_name,
      param_value_size,
      param_value,
      param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clBuildProgram(
    cl_program program,
    cl_uint num_devices,
    const cl_device_id *device_list,
    const char *options,
    void(CL_CALLBACK *pfn_notify)(cl_program program, void *user_data),
    void *user_data) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clBuildProgram()(
      program, num_devices, device_list, options, pfn_notify, user_data);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueNDRangeKernel(cl_command_queue command_queue,
                       cl_kernel kernel,
                       cl_uint work_dim,
                       const size_t *global_work_offset,
                       const size_t *global_work_size,
                       const size_t *local_work_size,
                       cl_uint num_events_in_wait_list,
                       const cl_event *event_wait_list,
                       cl_event *event) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clEnqueueNDRangeKernel()(
      command_queue,
      kernel,
      work_dim,
      global_work_offset,
      global_work_size,
      local_work_size,
      num_events_in_wait_list,
      event_wait_list,
      event);
}

CL_API_ENTRY cl_int CL_API_CALL clSetKernelArg(cl_kernel kernel,
                                               cl_uint arg_index,
                                               size_t arg_size,
                                               const void *arg_value)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clSetKernelArg()(
      kernel, arg_index, arg_size, arg_value);
}

CL_API_ENTRY cl_int CL_API_CALL clRetainMemObject(cl_mem memobj)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clRetainMemObject()(memobj);
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseMemObject(cl_mem memobj)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clReleaseMemObject()(memobj);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueUnmapMemObject(cl_command_queue command_queue,
                        cl_mem memobj,
                        void *mapped_ptr,
                        cl_uint num_events_in_wait_list,
                        const cl_event *event_wait_list,
                        cl_event *event) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clEnqueueUnmapMemObject()(
      command_queue,
      memobj,
      mapped_ptr,
      num_events_in_wait_list,
      event_wait_list,
      event);
}

CL_API_ENTRY cl_int CL_API_CALL clRetainCommandQueue(
    cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clRetainCommandQueue()(
      command_queue);
}

CL_API_ENTRY cl_context CL_API_CALL
clCreateContext(const cl_context_properties *properties,
                cl_uint num_devices,
                const cl_device_id *devices,
                void(CL_CALLBACK *pfn_notify)(const char *errinfo,
                                              const void *private_info,
                                              size_t cb,
                                              void *user_data),
                void *user_data,
                cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clCreateContext()(
      properties, num_devices, devices, pfn_notify, user_data, errcode_ret);
}

CL_API_ENTRY cl_context CL_API_CALL
clCreateContextFromType(const cl_context_properties *properties,
                        cl_device_type device_type,
                        void(CL_CALLBACK *pfn_notify)(const char *errinfo,
                                                      const void *private_info,
                                                      size_t cb,
                                                      void *user_data),
                        void *user_data,
                        cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clCreateContextFromType()(
      properties, device_type, pfn_notify, user_data, errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseContext(cl_context context)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clReleaseContext()(context);
}

CL_API_ENTRY cl_int CL_API_CALL clWaitForEvents(
    cl_uint num_events, const cl_event *event_list) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clWaitForEvents()(num_events,
                                                              event_list);
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseEvent(cl_event event)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clReleaseEvent()(event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteBuffer(cl_command_queue command_queue,
                     cl_mem buffer,
                     cl_bool blocking_write,
                     size_t offset,
                     size_t size,
                     const void *ptr,
                     cl_uint num_events_in_wait_list,
                     const cl_event *event_wait_list,
                     cl_event *event) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clEnqueueWriteBuffer()(
      command_queue,
      buffer,
      blocking_write,
      offset,
      size,
      ptr,
      num_events_in_wait_list,
      event_wait_list,
      event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadBuffer(cl_command_queue command_queue,
                    cl_mem buffer,
                    cl_bool blocking_read,
                    size_t offset,
                    size_t size,
                    void *ptr,
                    cl_uint num_events_in_wait_list,
                    const cl_event *event_wait_list,
                    cl_event *event) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clEnqueueReadBuffer()(
      command_queue,
      buffer,
      blocking_read,
      offset,
      size,
      ptr,
      num_events_in_wait_list,
      event_wait_list,
      event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueReadImage(cl_command_queue command_queue,
                   cl_mem image,
                   cl_bool blocking_read,
                   const size_t *origin,
                   const size_t *region,
                   size_t row_pitch,
                   size_t slice_pitch,
                   void *ptr,
                   cl_uint num_events_in_wait_list,
                   const cl_event *event_wait_list,
                   cl_event *event) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clEnqueueReadImage()(
      command_queue,
      image,
      blocking_read,
      origin,
      region,
      row_pitch,
      slice_pitch,
      ptr,
      num_events_in_wait_list,
      event_wait_list,
      event);
}

CL_API_ENTRY cl_int CL_API_CALL
clGetProgramBuildInfo(cl_program program,
                      cl_device_id device,
                      cl_program_build_info param_name,
                      size_t param_value_size,
                      void *param_value,
                      size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clGetProgramBuildInfo()(
      program,
      device,
      param_name,
      param_value_size,
      param_value,
      param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clRetainProgram(cl_program program)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clRetainProgram()(program);
}

CL_API_ENTRY void *CL_API_CALL
clEnqueueMapBuffer(cl_command_queue command_queue,
                   cl_mem buffer,
                   cl_bool blocking_map,
                   cl_map_flags map_flags,
                   size_t offset,
                   size_t size,
                   cl_uint num_events_in_wait_list,
                   const cl_event *event_wait_list,
                   cl_event *event,
                   cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clEnqueueMapBuffer()(
      command_queue,
      buffer,
      blocking_map,
      map_flags,
      offset,
      size,
      num_events_in_wait_list,
      event_wait_list,
      event,
      errcode_ret);
}

CL_API_ENTRY void *CL_API_CALL
clEnqueueMapImage(cl_command_queue command_queue,
                  cl_mem image,
                  cl_bool blocking_map,
                  cl_map_flags map_flags,
                  const size_t *origin,
                  const size_t *region,
                  size_t *image_row_pitch,
                  size_t *image_slice_pitch,
                  cl_uint num_events_in_wait_list,
                  const cl_event *event_wait_list,
                  cl_event *event,
                  cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clEnqueueMapImage()(
      command_queue,
      image,
      blocking_map,
      map_flags,
      origin,
      region,
      image_row_pitch,
      image_slice_pitch,
      num_events_in_wait_list,
      event_wait_list,
      event,
      errcode_ret);
}

CL_API_ENTRY CL_EXT_PREFIX__VERSION_1_2_DEPRECATED cl_command_queue CL_API_CALL
clCreateCommandQueue(cl_context context,
                     cl_device_id device,
                     cl_command_queue_properties properties,
                     cl_int *errcode_ret)
    CL_EXT_SUFFIX__VERSION_1_2_DEPRECATED {
  return paddle::lite::CLWrapper::Global()->clCreateCommandQueue()(
      context, device, properties, errcode_ret);
}

CL_API_ENTRY cl_command_queue CL_API_CALL clCreateCommandQueueWithProperties(
    cl_context context,
    cl_device_id device,
    const cl_queue_properties *properties,
    cl_int *errcode_ret) CL_API_SUFFIX__VERSION_2_0 {
  return paddle::lite::CLWrapper::Global()
      ->clCreateCommandQueueWithProperties()(
          context, device, properties, errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseCommandQueue(
    cl_command_queue command_queue) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clReleaseCommandQueue()(
      command_queue);
}

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithBinary(cl_context context,
                          cl_uint num_devices,
                          const cl_device_id *device_list,
                          const size_t *lengths,
                          const unsigned char **binaries,
                          cl_int *binary_status,
                          cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clCreateProgramWithBinary()(
      context,
      num_devices,
      device_list,
      lengths,
      binaries,
      binary_status,
      errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clRetainContext(cl_context context)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clRetainContext()(context);
}

CL_API_ENTRY cl_int CL_API_CALL clGetContextInfo(cl_context context,
                                                 cl_context_info param_name,
                                                 size_t param_value_size,
                                                 void *param_value,
                                                 size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clGetContextInfo()(
      context, param_name, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseProgram(cl_program program)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clReleaseProgram()(program);
}

CL_API_ENTRY cl_int CL_API_CALL clFlush(cl_command_queue command_queue)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clFlush()(command_queue);
}

CL_API_ENTRY cl_int CL_API_CALL clFinish(cl_command_queue command_queue)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clFinish()(command_queue);
}

CL_API_ENTRY cl_int CL_API_CALL clGetProgramInfo(cl_program program,
                                                 cl_program_info param_name,
                                                 size_t param_value_size,
                                                 void *param_value,
                                                 size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clGetProgramInfo()(
      program, param_name, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_kernel CL_API_CALL clCreateKernel(cl_program program,
                                                  const char *kernel_name,
                                                  cl_int *errcode_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clCreateKernel()(
      program, kernel_name, errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clRetainKernel(cl_kernel kernel)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clRetainKernel()(kernel);
}

CL_API_ENTRY cl_mem CL_API_CALL clCreateBuffer(cl_context context,
                                               cl_mem_flags flags,
                                               size_t size,
                                               void *host_ptr,
                                               cl_int *errcode_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clCreateBuffer()(
      context, flags, size, host_ptr, errcode_ret);
}

CL_API_ENTRY CL_EXT_PREFIX__VERSION_1_1_DEPRECATED cl_mem CL_API_CALL
clCreateImage2D(cl_context context,
                cl_mem_flags flags,
                const cl_image_format *image_format,
                size_t image_width,
                size_t image_height,
                size_t image_row_pitch,
                void *host_ptr,
                cl_int *errcode_ret) CL_EXT_SUFFIX__VERSION_1_1_DEPRECATED {
  return paddle::lite::CLWrapper::Global()->clCreateImage2D()(context,
                                                              flags,
                                                              image_format,
                                                              image_width,
                                                              image_height,
                                                              image_row_pitch,
                                                              host_ptr,
                                                              errcode_ret);
}

CL_API_ENTRY cl_mem CL_API_CALL
clCreateImage(cl_context context,
              cl_mem_flags flags,
              const cl_image_format *image_format,
              const cl_image_desc *image_desc,
              void *host_ptr,
              cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_2 {
  return paddle::lite::CLWrapper::Global()->clCreateImage()(
      context, flags, image_format, image_desc, host_ptr, errcode_ret);
}

CL_API_ENTRY cl_event CL_API_CALL clCreateUserEvent(
    cl_context context, cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_1 {
  return paddle::lite::CLWrapper::Global()->clCreateUserEvent()(context,
                                                                errcode_ret);
}

CL_API_ENTRY cl_program CL_API_CALL
clCreateProgramWithSource(cl_context context,
                          cl_uint count,
                          const char **strings,
                          const size_t *lengths,
                          cl_int *errcode_ret) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clCreateProgramWithSource()(
      context, count, strings, lengths, errcode_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseKernel(cl_kernel kernel)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clReleaseKernel()(kernel);
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceInfo(cl_device_id device,
                                                cl_device_info param_name,
                                                size_t param_value_size,
                                                void *param_value,
                                                size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clGetDeviceInfo()(
      device, param_name, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetDeviceIDs(cl_platform_id platform,
                                               cl_device_type device_type,
                                               cl_uint num_entries,
                                               cl_device_id *devices,
                                               cl_uint *num_devices)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clGetDeviceIDs()(
      platform, device_type, num_entries, devices, num_devices);
}

CL_API_ENTRY cl_int CL_API_CALL clRetainDevice(cl_device_id device)
    CL_API_SUFFIX__VERSION_1_2 {
  return paddle::lite::CLWrapper::Global()->clRetainDevice()(device);
}

CL_API_ENTRY cl_int CL_API_CALL clReleaseDevice(cl_device_id device)
    CL_API_SUFFIX__VERSION_1_2 {
  return paddle::lite::CLWrapper::Global()->clReleaseDevice()(device);
}

CL_API_ENTRY cl_int CL_API_CALL clRetainEvent(cl_event event)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clRetainEvent()(event);
}

CL_API_ENTRY cl_int CL_API_CALL clGetKernelWorkGroupInfo(
    cl_kernel kernel,
    cl_device_id device,
    cl_kernel_work_group_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clGetKernelWorkGroupInfo()(
      kernel,
      device,
      param_name,
      param_value_size,
      param_value,
      param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetEventInfo(cl_event event,
                                               cl_event_info param_name,
                                               size_t param_value_size,
                                               void *param_value,
                                               size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clGetEventInfo()(
      event, param_name, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetEventProfilingInfo(
    cl_event event,
    cl_profiling_info param_name,
    size_t param_value_size,
    void *param_value,
    size_t *param_value_size_ret) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clGetEventProfilingInfo()(
      event, param_name, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL clGetImageInfo(cl_mem image,
                                               cl_image_info param_name,
                                               size_t param_value_size,
                                               void *param_value,
                                               size_t *param_value_size_ret)
    CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clGetImageInfo()(
      image, param_name, param_value_size, param_value, param_value_size_ret);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyBuffer(cl_command_queue command_queue,
                    cl_mem src_buffer,
                    cl_mem dst_buffer,
                    size_t src_offset,
                    size_t dst_offset,
                    size_t size,
                    cl_uint num_events_in_wait_list,
                    const cl_event *event_wait_list,
                    cl_event *event) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clEnqueueCopyBuffer()(
      command_queue,
      src_buffer,
      dst_buffer,
      src_offset,
      dst_offset,
      size,
      num_events_in_wait_list,
      event_wait_list,
      event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueWriteImage(cl_command_queue command_queue,
                    cl_mem image,
                    cl_bool blocking_write,
                    const size_t *origin,
                    const size_t *region,
                    size_t input_row_pitch,
                    size_t input_slice_pitch,
                    const void *ptr,
                    cl_uint num_events_in_wait_list,
                    const cl_event *event_wait_list,
                    cl_event *event) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clEnqueueWriteImage()(
      command_queue,
      image,
      blocking_write,
      origin,
      region,
      input_row_pitch,
      input_slice_pitch,
      ptr,
      num_events_in_wait_list,
      event_wait_list,
      event);
}

CL_API_ENTRY cl_int CL_API_CALL
clEnqueueCopyImage(cl_command_queue command_queue,
                   cl_mem src_image,
                   cl_mem dst_image,
                   const size_t *src_origin,
                   const size_t *dst_origin,
                   const size_t *region,
                   cl_uint num_events_in_wait_list,
                   const cl_event *event_wait_list,
                   cl_event *event) CL_API_SUFFIX__VERSION_1_0 {
  return paddle::lite::CLWrapper::Global()->clEnqueueCopyImage()(
      command_queue,
      src_image,
      dst_image,
      src_origin,
      dst_origin,
      region,
      num_events_in_wait_list,
      event_wait_list,
      event);
}
