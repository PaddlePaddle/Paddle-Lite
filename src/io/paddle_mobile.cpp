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

#include "io/paddle_mobile.h"
#include <utility>
#include "common/common.h"
#ifdef _OPENMP
#include <omp.h>
#endif  // _OPENMP
#ifdef PADDLE_MOBILE_CL
#include <CL/cl.h>
#include <mutex>
#include "framework/cl/cl_engine.h"
#include "framework/cl/cl_tensor.h"
#endif
#include "operators/math/gemm.h"

namespace paddle_mobile {

template <typename Device, typename T>
void PaddleMobile<Device, T>::SetThreadNum(int num) {
  executor_->SetThreadNum(num);
}

template <typename Device, typename T>
PMStatus PaddleMobile<Device, T>::Load(const std::string &dirname,
                                       bool optimize, bool quantification,
                                       int batch_size, bool lod_mode) {
  if (loader_.get() == nullptr) {
    loader_ = std::make_shared<framework::Loader<Device, T>>();
  } else {
    LOG(kLOG_INFO) << "loader inited";
  }

  if (executor_.get() == nullptr) {
    executor_ = std::make_shared<framework::Executor<Device, T>>(
        loader_->Load(dirname, optimize, quantification), config_, batch_size,
        optimize, lod_mode);
  } else {
    LOG(kLOG_INFO) << "executor inited";
  }

  return PMSuccess;
}

template <typename Device, typename T>
PMStatus PaddleMobile<Device, T>::Load(const std::string &model_path,
                                       const std::string &para_path,
                                       bool optimize, bool quantification,
                                       int batch_size, bool lod_mode) {
  if (loader_.get() == nullptr) {
    loader_ = std::make_shared<framework::Loader<Device, T>>();
  } else {
    LOG(kLOG_INFO) << "loader inited";
  }

  if (executor_.get() == nullptr) {
    executor_ = std::make_shared<framework::Executor<Device, T>>(
        loader_->Load(model_path, para_path, optimize, quantification), config_,
        batch_size, optimize, lod_mode);
  } else {
    LOG(kLOG_INFO) << "executor inited";
  }

  return PMSuccess;
}

template <typename Device, typename T>
PMStatus PaddleMobile<Device, T>::Load(const PaddleMobileConfig &config) {
  if (!config.model_dir.empty()) {
    return this->Load(config.model_dir, config.optimize, config.quantification,
                      config.batch_size, config.lod_mode);
  } else if (!config.prog_file.empty() && !config.param_file.empty()) {
    return this->Load(config.prog_file, config.param_file, config.optimize,
                      config.quantification, config.batch_size,
                      config.lod_mode);
  } else {
    LOG(kLOG_ERROR) << "Failed to load inference model";
    return PMNotInitialized;
  }
}

template <typename Device, typename T>
bool PaddleMobile<Device, T>::LoadCombinedMemory(
    size_t model_len, const uint8_t *model_buf, size_t combined_params_len,
    uint8_t *combined_params_buf, bool optimize, bool quantification,
    int batch_size, bool lod_mode) {
  if (loader_.get() == nullptr) {
    loader_ = std::make_shared<framework::Loader<Device, T>>();
  } else {
    LOG(kLOG_INFO) << "loader inited";
  }
  if (executor_.get() == nullptr) {
    executor_ = std::make_shared<framework::Executor<Device, T>>(
        loader_->LoadCombinedMemory(model_len, model_buf, combined_params_len,
                                    combined_params_buf, optimize,
                                    quantification),
        config_, batch_size, optimize, lod_mode);
  } else {
    LOG(kLOG_INFO) << "executor inited";
  }

  return PMSuccess;
}

template <typename Device, typename T>
PMStatus PaddleMobile<Device, T>::Predict(const framework::Tensor &input) {
  std::vector<std::pair<std::string, framework::Tensor>> inputs;
  inputs.push_back(std::make_pair("feed", input));
  return this->Predict(inputs);
}

template <typename Device, typename T>
PMStatus PaddleMobile<Device, T>::Predict(const framework::LoDTensor &input) {
  std::vector<std::pair<std::string, framework::LoDTensor>> inputs;
  inputs.push_back(std::make_pair("feed", input));
  return this->Predict(inputs);
}

template <typename Device, typename T>
PMStatus PaddleMobile<Device, T>::Predict(
    const std::vector<std::pair<std::string, framework::Tensor>> &inputs) {
  return executor_->Predict(inputs);
}

template <typename Device, typename T>
PMStatus PaddleMobile<Device, T>::Predict(
    const std::vector<std::pair<std::string, framework::LoDTensor>> &inputs) {
  return executor_->Predict(inputs);
}

template <typename Device, typename T>
std::vector<T> PaddleMobile<Device, T>::Predict(
    const std::vector<T> &input, const std::vector<int64_t> &dims) {
  return executor_->Predict(input, dims);
}

template <typename Device, typename T>
PMStatus PaddleMobile<Device, T>::Predict() {
  return executor_->Predict();
}

template <typename Device, typename T>
void PaddleMobile<Device, T>::Feed(const std::string &var_name,
                                   const framework::Tensor &input) {
  executor_->SetInput(input, var_name);
}

template <typename Device, typename T>
void PaddleMobile<Device, T>::Feed(const std::string &var_name,
                                   const framework::LoDTensor &input) {
  executor_->SetInput(input, var_name);
}

typedef std::shared_ptr<framework::LoDTensor> LoDTensorPtr;
template <typename Device, typename T>
LoDTensorPtr PaddleMobile<Device, T>::Fetch(const std::string &var_name) {
  return executor_->GetOutput(var_name);
}

template <typename Device, typename T>
void PaddleMobile<Device, T>::Clear() {
  executor_ = nullptr;
  loader_ = nullptr;
}

template <typename Device, typename T>
double PaddleMobile<Device, T>::GetPredictTime() {return 0;}

#ifdef PADDLE_MOBILE_CPU
template <>
double PaddleMobile<CPU, float>::GetPredictTime() {
  int m = 32;
  int n = 224 * 224;
  int k = 27;
  int lda = k;
  int ldb = n;
  int ldc = n;
  float *a =
      static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * m * k));
  float *b =
      static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * k * n));
  float *c =
      static_cast<float *>(paddle_mobile::memory::Alloc(sizeof(float) * m * n));
  int t1 = 1;
  int t2 = 1;
  for (int i = 0; i < m * k; ++i) {
    a[i] = t1 + rand() % t2;  // NOLINT
  }
  for (int i = 0; i < k * n; ++i) {
    b[i] = t1 + rand() % t2;  // NOLINT
  }

  operators::math::Gemm gemm;
  auto time1 = paddle_mobile::time();
  int times = 4;
  for (int j = 0; j < times; ++j) {
    gemm.Sgemm(m, n, k, static_cast<float>(1), a, lda, b, ldb,
               static_cast<float>(0), c, ldc, false,
               static_cast<float *>(nullptr));
  }

  auto time2 = paddle_mobile::time();
  double cost = paddle_mobile::time_diff(time1, time2) / times;
  paddle_mobile::memory::Free(a);
  paddle_mobile::memory::Free(b);
  paddle_mobile::memory::Free(c);
  return cost;
}
#endif

#ifdef PADDLE_MOBILE_FPGA
template <typename Device, typename T>
void PaddleMobile<Device, T>::InjectVariable(const framework::Tensor &t,
                                             std::string var_name) {
  executor_->InjectVariable(t, var_name);
}

template <typename Device, typename T>
void PaddleMobile<Device, T>::FeedData(const framework::Tensor &t) {
  executor_->FeedData(t);
}

template <typename Device, typename T>
void PaddleMobile<Device, T>::FeedData(const std::vector<void *> &v) {
  executor_->FeedData(v);
}
template <typename Device, typename T>
void PaddleMobile<Device, T>::FeedTensorData(
    const std::vector<framework::Tensor> &v) {
  executor_->FeedTensorData(v);
}

template <typename Device, typename T>
void PaddleMobile<Device, T>::GetResults(std::vector<void *> *v) {
  executor_->GetResults(v);
}

template <typename Device, typename T>
void PaddleMobile<Device, T>::GetTensorResults(
    std::vector<framework::Tensor *> *v) {
  executor_->GetTensorResults(v);
}

template <typename Device, typename T>
framework::Tensor *PaddleMobile<Device, T>::GetTensorByName(
    const std::string &name) {
  return executor_->GetTensorByName(name);
}

template <typename Device, typename T>
std::shared_ptr<framework::Tensor> PaddleMobile<Device, T>::FetchResult(
    int id) {
  return executor_->FetchResult(id);
}

template <typename Device, typename T>
void PaddleMobile<Device, T>::Predict_From_To(int start, int end) {
  executor_->Predict_From_To(start, end);
}

template <typename Device, typename T>
void PaddleMobile<Device, T>::Predict_From(int start) {
  executor_->Predict_From(start);
}

template <typename Device, typename T>
void PaddleMobile<Device, T>::Predict_To(int end) {
  executor_->Predict_To(end);
}
#endif

#ifdef PADDLE_MOBILE_CL
static std::mutex lc;
template <typename Device, typename T>
void PaddleMobile<Device, T>::SetCLPath(std::string path) {
  std::lock_guard<std::mutex> lock(lc);
  if (framework::CLEngine::Instance()->GetCLPath() == "") {
    framework::CLEngine::Instance()->setClPath(path);
  }
}
template <>
double PaddleMobile<GPU_CL, float>::GetPredictTime() {
  cl_int status;
  if (!framework::CLEngine::Instance()->isInitSuccess()) {
    return -1;
  }
  cl_context context = framework::CLEngine::Instance()->getContext();
  cl_command_queue queue = framework::CLEngine::Instance()->getClCommandQueue();

  int n = 1;
  int c = 3;
  int h = 224;
  int w = 224;
  float *input = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * 3 * 224 * 224));
  float *filter = static_cast<float *>(
      paddle_mobile::memory::Alloc(sizeof(float) * 32 * 27));
  int input_w = w * (c + 3) / 4;
  int input_h = n * h;
  int filter_w = 3 * (3 + 3) / 4;
  int filter_h = 32 * 3;
  int output_w = 224 * (32 + 3) / 4;
  int output_h = 1 * 224;

  framework::DDim input_dims = {1, 3, 224, 224};
  framework::CLTensor input_cl_tensor(context, queue);
  input_cl_tensor.Resize(input_dims);
  cl_mem inputBuffer = input_cl_tensor.mutable_with_data<float>(input);

  framework::DDim filter_dims = {32, 3, 3, 3};
  framework::CLTensor filter_cl_tensor(context, queue);
  input_cl_tensor.Resize(filter_dims);
  cl_mem filterBuffer = filter_cl_tensor.mutable_with_data<float>(filter);

  cl_mem cl_filter_image = NULL;
  cl_mem cl_input_image = NULL;
  cl_mem cl_output_image = NULL;
  cl_image_format cf = {.image_channel_order = CL_RGBA,
                        .image_channel_data_type = CL_HALF_FLOAT};
  cl_input_image = clCreateImage2D(context, CL_MEM_READ_WRITE | 0, &cf, input_w,
                                   input_h, 0, NULL, &status);
  cl_filter_image = clCreateImage2D(context, CL_MEM_READ_WRITE | 0, &cf,
                                    filter_w, filter_h, 0, NULL, &status);
  cl_output_image = clCreateImage2D(context, CL_MEM_READ_WRITE | 0, &cf,
                                    output_w, output_h, 0, NULL, &status);
  char *code;
  std::string path = framework::CLEngine::Instance()->GetCLPath() +
                     "/cl_kernel/feed_kernel.cl";
  size_t length = readText(path.c_str(), &code);
  cl_program program = clCreateProgramWithSource(
      context, 1, (const char **)&code, &length, NULL);
  std::string path1 = "-cl-fast-relaxed-math -I " +
                      framework::CLEngine::Instance()->GetCLPath() +
                      "/cl_kernel";
  clBuildProgram(program, 0, 0, path1.c_str(), NULL, NULL);
  cl_kernel kernel = clCreateKernel(program, "feed", &status);

  int out_H = 224;
  int out_W = 224;
  int out_C = 3;
  int Stride2 = out_C * out_H * out_W;
  int Stride1 = out_H * out_W;
  int Stride0 = out_W;
  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuffer);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_input_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(cl_int), &out_H);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_int), &out_W);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(cl_int), &out_C);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(cl_int), &Stride0);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(cl_int), &Stride1);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(cl_int), &Stride2);
  CL_CHECK_ERRORS(status);

  size_t global_work_size[3] = {1, 224, 224};

  //  cl_event out_event = param.Out()->GetClEvent();

  status = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size,
                                  NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);

  out_H = 3;
  out_W = 3;
  out_C = 3;
  Stride2 = out_C * out_H * out_W;
  Stride1 = out_H * out_W;
  Stride0 = out_W;

  status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &filterBuffer);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(cl_mem), &cl_filter_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(cl_int), &out_H);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_int), &out_W);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(cl_int), &out_C);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(cl_int), &Stride0);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(cl_int), &Stride1);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(cl_int), &Stride2);
  CL_CHECK_ERRORS(status);

  size_t global_work_size1[3] = {1, 3, 96};

  //  cl_event out_event = param.Out()->GetClEvent();

  status = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size1,
                                  NULL, 0, NULL, NULL);
  CL_CHECK_ERRORS(status);

  clFinish(queue);
  //  queue = clCreateCommandQueue(context, listDevice[0], 0, &status);

  path = framework::CLEngine::Instance()->GetCLPath() +
         "/cl_kernel/conv_kernel.cl";
  size_t length1 = readText(path.c_str(), &code);
  program = clCreateProgramWithSource(context, 1, (const char **)&code,
                                      &length1, &status);
  CL_CHECK_ERRORS(status);
  clBuildProgram(program, 0, 0, path1.c_str(), NULL, NULL);
  kernel = clCreateKernel(program, "conv_3x3", &status);
  CL_CHECK_ERRORS(status);

  int c_block = (32 + 3) / 4;
  int nh = n * h;
  int stride = 1;
  int offset = 0;
  int input_c = (c + 3) / 4;
  int dilation = 1;
  int input_width = 224;
  int input_height = 224;
  int output_width = 224;
  int output_height = 224;
  status = clSetKernelArg(kernel, 0, sizeof(int), &c_block);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 1, sizeof(int), &w);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 2, sizeof(int), &nh);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 3, sizeof(cl_mem), &cl_input_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 4, sizeof(cl_mem), &cl_filter_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 5, sizeof(cl_mem), &cl_output_image);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 6, sizeof(int), &stride);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 7, sizeof(int), &offset);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 8, sizeof(int), &input_c);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 9, sizeof(int), &dilation);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 10, sizeof(int), &input_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 11, sizeof(int), &input_height);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 12, sizeof(int), &output_width);
  CL_CHECK_ERRORS(status);
  status = clSetKernelArg(kernel, 13, sizeof(int), &output_height);
  CL_CHECK_ERRORS(status);

  //  cl_event out_event = param.Output()->GetClEvent();
  //  cl_event wait_event = param.Input()->GetClEvent();
  size_t global_work_size2[3] = {8, 224, 224};
  auto time1 = paddle_mobile::time();
  int times = 10;
  for (int i = 0; i < times; ++i) {
    status = clEnqueueNDRangeKernel(queue, kernel, 3, NULL, global_work_size2,
                                    NULL, 0, NULL, NULL);
  }
  CL_CHECK_ERRORS(status);
  clFinish(queue);
  auto time2 = paddle_mobile::time();
  paddle_mobile::memory::Free(input);
  paddle_mobile::memory::Free(filter);
  if (status == CL_SUCCESS) {
    return paddle_mobile::time_diff(time1, time2) / times;
  } else {
    return -1;
  }
}
template <typename Device, typename T>
int PaddleMobile<Device, T>::readText(
    const char *kernelPath,
    char **pcode) {  // 读取文本文件放入 pcode，返回字符串长度
  FILE *fp;
  int size;
  // printf("<readText> File: %s\n", kernelPath);
  fp = fopen(kernelPath, "rb");
  if (!fp) {
    printf("<readText> Open file failed\n");
    return -1;
  }
  if (fseek(fp, 0, SEEK_END) != 0) {
    printf("<readText> Seek end of file failed\n");
    return -1;
  }
  if ((size = ftell(fp)) < 0) {
    printf("<readText> Get file position failed\n");
    return -1;
  }
  rewind(fp);
  if ((*pcode = reinterpret_cast<char *>(malloc(size + 1))) == NULL) {
    printf("<readText> Allocate space failed\n");
    return -1;
  }
  fread(*pcode, 1, size, fp);
  (*pcode)[size] = '\0';
  fclose(fp);
  return size + 1;
}
#endif

template class PaddleMobile<CPU, float>;
template class PaddleMobile<FPGA, float>;
template class PaddleMobile<GPU_MALI, float>;
template class PaddleMobile<GPU_CL, float>;

}  // namespace paddle_mobile
