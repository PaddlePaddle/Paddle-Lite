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

#include <vector>
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/replace_stl/stream.h"
#include "lite/utils/string.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/host/math/split.h"
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {

class RnnCompute
    : public KernelLite<TARGET(kOpenCL), PRECISION(kFloat), DATALAYOUT(kNCHW)> {
 public:
  using param_t = operators::RnnParam;

  void TransW(const float* src, float* dst, int n, int k) {
    if (src == nullptr || dst == nullptr) return;
    for (int i = 0; i < k; i++) {
      for (int j = 0; j < n; j++) {
        dst[i * n + j] = src[j * k + i];
        if (i == 0 && j == 0)
          std::cout << "TransW i:" << i << "; j:" << j << "->" << dst[i * n + j]
                    << std::endl;
      }
    }
  }

  void reset_parameter_vector(const std::vector<Tensor*>& raw_params_vec,
                              const int& num_layers,
                              const int& gate_num,
                              const bool& is_bidirec,
                              std::vector<std::vector<Tensor>>* params_vec) {
    // the parameter raw seuquence is [FWhi, FWhh, BWhi, BWhh] * num_layers
    // + [FBhi, FBhh, BBhi, BBhh] * num_layers, we will reset the parameter to
    // ([FWhi, FWhh, FBhi, FBhh] + [BWhi, BWhh, BBhi, BBhh]) * num_layers
    const int& direction_num = is_bidirec ? 2 : 1;
    const int& layer_weight_size = 4 * direction_num;
    const int& all_weight_size = num_layers * layer_weight_size;
    const int& bias_start_idx = all_weight_size / 2;
    for (int i = 0; i < num_layers; i++) {
      std::vector<Tensor> tensor_list;
      tensor_list.reserve(layer_weight_size);
      for (int j = 0; j < layer_weight_size; j++) {
        Tensor tensor_holder;
        tensor_list.emplace_back(tensor_holder);
      }
      for (int j = 0; j < layer_weight_size; j++) {
        int k = j % 4;
        const int& section = j / 4;
        int tensor_idx = i * 2 * direction_num + section * 2 + k % 2;
        if (k >= 2) {
          tensor_idx += bias_start_idx;
        }
        tensor_list[j].ShareDataWith(*raw_params_vec[tensor_idx]);
      }
      params_vec->emplace_back(tensor_list);
    }
  }

  void SwapPoniter(Tensor** a, Tensor** b) {
    Tensor* c = *a;
    *a = *b;
    *b = c;
  }

  void preprocess(const Tensor* input,
                  const Tensor& weight,
                  const Tensor& bias_ih,
                  const Tensor& bias_hh,
                  std::string mode,
                  Tensor* cache_input) {
    std::cout << "preprocess~~~" << std::endl;
    const int& hidden_size = weight.dims()[0];
    int time_step = input->dims()[0];
    int batch = input->dims()[1];

    auto input_dims = input->dims();
    auto weight_input_dims = weight.dims();
    int m = input_dims[0] * input_dims[1];
    int k = input_dims[2];
    int n = weight_input_dims[0];

    std::vector<int64_t> cache_input_dim = {
        time_step, batch, hidden_size};  // 60 1 2048
    DDim gate_dim;
    gate_dim.ConstructFrom(cache_input_dim);
    cache_input->Resize(gate_dim);

    auto* i_buf = GET_BUFFER_GPU(input);
    auto* out_buf =
        (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
            ? cache_input->mutable_data<half_t, cl::Buffer>(TARGET(kOpenCL))
            : cache_input->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

    auto* vec0_img = DATA_GPU(vec0_gpu_t_);
    auto* vec2_img = DATA_GPU(vec2_gpu_t_);
    auto* vec3_img = DATA_GPU(vec3_gpu_t_);
    std::cout << "preprocess_kernel_ m n k " << m << ", " << n << ", " << k
              << std::endl;
    cl_int status;
    auto kernel = preprocess_kernel_;
    status = kernel.setArg(0, *i_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(1, *vec0_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(2, *vec2_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(3, *vec3_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(4, *out_buf);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(5, static_cast<const int>(m));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(6, static_cast<const int>(n));
    CL_CHECK_FATAL(status);
    status = kernel.setArg(7, static_cast<const int>(k));
    CL_CHECK_FATAL(status);
    auto& context = ctx_->As<OpenCLContext>();
    status = EnqueueNDRangeKernel(context,
                                  preprocess_kernel_,
                                  cl::NullRange,
                                  cl::NDRange(static_cast<size_t>((m + 3) / 4),
                                              static_cast<size_t>((n + 3) / 4)),
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
    event_.wait();
    auto queue_start_nanos =
        event_.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
    auto submit_start_nanos =
        event_.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
    auto run_start_nanos =
        event_.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    auto run_stop_nanos = event_.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    double time_ms = (submit_start_nanos - queue_start_nanos) / 1000000.0;
    std::cout << "preprocess GetQueuedToSubmitTime: " << time_ms << std::endl;

    time_ms = (run_start_nanos - submit_start_nanos) / 1000000.0;
    std::cout << "preprocess GetSubmitToStartTime: " << time_ms << std::endl;

    time_ms = (run_stop_nanos - run_start_nanos) / 1000000.0;
    std::cout << "preprocess GetStartToEndTime: " << time_ms << std::endl;
  }

  void lstm_cell(Tensor* input,  // 15*2048
                 Tensor* weight_hh,
                 Tensor* init_h,
                 Tensor* last_c_act,
                 Tensor* output,  // 15*2048
                 const Tensor* bias_hh,
                 int time_step_id,
                 std::vector<Tensor*> state,
                 int max_step) {
    auto& context = ctx_->As<OpenCLContext>();
    auto* output_buf =
        (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
            ? output->mutable_data<half_t, cl::Buffer>(TARGET(kOpenCL))
            : output->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

    auto* output_buf1 =
        (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
            ? state[0]->mutable_data<half_t, cl::Buffer>(TARGET(kOpenCL))
            : state[0]->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

    auto* output_buf2 =
        (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
            ? state[1]->mutable_data<half_t, cl::Buffer>(TARGET(kOpenCL))
            : state[1]->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

    size_t frame_size = init_h->dims()[1];
    float cell_clip = 0.0;
    // init_h weight_hh
    auto h_dims = init_h->dims();
    auto weight_input_dims = weight_hh->dims();
    int m = h_dims[0];
    int k = h_dims[1];
    int n = weight_input_dims[0];

    Tensor tmp_gate;
    tmp_gate.Resize({m, n});  //
    auto* out_buf =
        (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
            ? tmp_gate.mutable_data<half_t, cl::Buffer>(TARGET(kOpenCL))
            : tmp_gate.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
    auto* vec1_img = DATA_GPU(vec1_gpu_t_);
    auto* i_data = GET_BUFFER_GPU(input);
    auto* init_h_buf = GET_BUFFER_GPU(init_h_buf_t_);
    cl_int status;
    if (time_step_id == 0) {
      status = lstm_gemm_kernel_.setArg(0, *init_h_buf);  // 1 512
      CL_CHECK_FATAL(status);
    } else {
      status = lstm_gemm_kernel_.setArg(0, *output_buf);  // 15 512
      CL_CHECK_FATAL(status);
    }
    status = lstm_gemm_kernel_.setArg(1, *vec1_img);  // 512, 2048   2048, 512?
    CL_CHECK_FATAL(status);
    status = lstm_gemm_kernel_.setArg(2, *i_data);  // bias //15, 1, 2048
    CL_CHECK_FATAL(status);
    status = lstm_gemm_kernel_.setArg(3, *out_buf);
    CL_CHECK_FATAL(status);
    status = lstm_gemm_kernel_.setArg(4, static_cast<const int>(m));
    CL_CHECK_FATAL(status);
    status = lstm_gemm_kernel_.setArg(5, static_cast<const int>(n));
    CL_CHECK_FATAL(status);
    status = lstm_gemm_kernel_.setArg(6, static_cast<const int>(k));
    CL_CHECK_FATAL(status);
    status = lstm_gemm_kernel_.setArg(7, static_cast<const int>(time_step_id));
    CL_CHECK_FATAL(status);
    status = lstm_gemm_kernel_.setArg(8, static_cast<int>(frame_size));
    CL_CHECK_FATAL(status);
    status = EnqueueNDRangeKernel(context,
                                  lstm_gemm_kernel_,
                                  cl::NullRange,
                                  cl::NDRange{static_cast<size_t>((n + 3) / 4)},
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);

    event_.wait();
    auto queue_start_nanos =
        event_.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
    auto submit_start_nanos =
        event_.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
    auto run_start_nanos =
        event_.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    auto run_stop_nanos = event_.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    double time_ms = (submit_start_nanos - queue_start_nanos) / 1000000.0;
    std::cout << "lstm gemm GetQueuedToSubmitTime: " << time_ms << std::endl;

    time_ms = (run_start_nanos - submit_start_nanos) / 1000000.0;
    std::cout << "lstm gemm GetSubmitToStartTime: " << time_ms << std::endl;

    time_ms = (run_stop_nanos - run_start_nanos) / 1000000.0;
    std::cout << "lstm gemm GetStartToEndTime: " << time_ms << std::endl;

    Tensor cell_pre_act;
    if (last_c_act == nullptr) {
      cell_pre_act.Resize(init_h->dims());
      auto* cell_pre_act_buf =
          (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
              ? cell_pre_act.mutable_data<half_t, cl::Buffer>(TARGET(kOpenCL))
              : cell_pre_act.mutable_data<float, cl::Buffer>(TARGET(kOpenCL));
      last_c_act = &cell_pre_act;
    }
    auto* last_c_act_buf = GET_BUFFER_GPU(last_c_act);

    auto* last_c_buf = GET_BUFFER_GPU(last_c_buf_t_);
    auto* init_c_buf = GET_BUFFER_GPU(init_c_buf_t_);
    if (time_step_id % 2 == 0) {
      status = lstm_compute_kernel_.setArg(0, *out_buf);  // gate_value
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(1, *last_c_buf);  // state
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(
          2, *last_c_act_buf);  // state_active_value
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(3, *init_c_buf);  // prev_state_value
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(4, *output_buf);  // output
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(5, static_cast<int>(frame_size));
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(6, static_cast<float>(cell_clip));
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(7, static_cast<int>(time_step_id));
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(8, *output_buf1);
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(9, *output_buf2);
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(10, static_cast<int>(max_step));
      CL_CHECK_FATAL(status);
    } else {
      status = lstm_compute_kernel_.setArg(0, *out_buf);  // gate_value
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(1, *init_c_buf);  // state
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(
          2, *last_c_act_buf);  // state_active_value
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(3, *last_c_buf);  // prev_state_value
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(4, *output_buf);  // output
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(5, static_cast<int>(frame_size));
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(6, static_cast<float>(cell_clip));
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(7, static_cast<int>(time_step_id));
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(8, *output_buf1);
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(9, *output_buf2);
      CL_CHECK_FATAL(status);
      status = lstm_compute_kernel_.setArg(10, static_cast<int>(max_step));
      CL_CHECK_FATAL(status);
    }

    status = EnqueueNDRangeKernel(context,
                                  lstm_compute_kernel_,
                                  cl::NullRange,
                                  cl::NDRange(static_cast<size_t>(frame_size)),
                                  cl::NullRange,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
    event_.wait();
    queue_start_nanos = event_.getProfilingInfo<CL_PROFILING_COMMAND_QUEUED>();
    submit_start_nanos = event_.getProfilingInfo<CL_PROFILING_COMMAND_SUBMIT>();
    run_start_nanos = event_.getProfilingInfo<CL_PROFILING_COMMAND_START>();
    run_stop_nanos = event_.getProfilingInfo<CL_PROFILING_COMMAND_END>();

    time_ms = (submit_start_nanos - queue_start_nanos) / 1000000.0;
    std::cout << "lstm compute GetQueuedToSubmitTime: " << time_ms << std::endl;

    time_ms = (run_start_nanos - submit_start_nanos) / 1000000.0;
    std::cout << "lstm compute GetSubmitToStartTime: " << time_ms << std::endl;

    time_ms = (run_stop_nanos - run_start_nanos) / 1000000.0;
    std::cout << "lstm compute GetStartToEndTime: " << time_ms << std::endl;
  }

  void RunRnnLayer(const Tensor* input,
                   std::vector<Tensor> vec,
                   std::vector<Tensor> init_h,
                   std::vector<Tensor> init_c,
                   const Tensor* sequence_length,
                   std::vector<Tensor>* last_h_ptr,
                   std::vector<Tensor>* last_c_ptr,
                   Tensor* output,
                   int layer_idx,
                   Tensor* gate_value,
                   bool is_bidirect,  // false
                   int offset,        // 0
                   std::string mode,
                   std::vector<Tensor*> state) {
    std::cout << "RunRnnLayer~~~" << std::endl;
    bool is_reverse = false;
    if (is_bidirect) {
      layer_idx = 2 * layer_idx + offset;
      if (offset > 0) {
        is_reverse = true;
      }
    }

    const int& time_step = input->dims()[0];
    preprocess(input,
               vec[0 + offset * 4],  // weight
               vec[2 + offset * 4],  // bias_ih
               vec[3 + offset * 4],  // bias_hh
               mode,
               gate_value);  // 15, 2048

    Tensor mask_matrix;
    std::vector<Tensor> mask_vec;
    std::vector<Tensor*> mask_tensor_list;
    int mask_min_length = time_step;
    if (is_reverse) {
      mask_min_length = mask_min_length - time_step + 1;
    }
    Tensor init_h_temp;
    init_h_temp.Resize(init_h[layer_idx].dims());
    init_h_temp.CopyDataFrom(init_h[layer_idx]);
    Tensor* init_h_holder = &init_h_temp;

    last_c_buf_t_ = new Tensor;
    last_c_buf_t_->Resize(init_h[layer_idx].dims());
    auto* last_c_buf_t_buf =
        (CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16)
            ? last_c_buf_t_->mutable_data<half_t, cl::Buffer>(TARGET(kOpenCL))
            : last_c_buf_t_->mutable_data<float, cl::Buffer>(TARGET(kOpenCL));

    bool fp16_support =
        CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;

    if (fp16_support) {
      init_c_buf_t_ = new Tensor;
      const auto init_c_dims = init_c[layer_idx].dims();
      auto* init_c_cpu = init_c[layer_idx].mutable_data<float>();
      auto init_c_cpu_t = std::unique_ptr<Tensor>(new Tensor);
      init_c_cpu_t->Resize(init_c_dims);
      init_c_buf_t_->Resize(init_c_dims);
      auto* init_c_buffer_data = MUTABLE_DATA_CPU(init_c_cpu_t.get());
      FloatArray2HalfArray(static_cast<float*>(init_c_cpu),
                           static_cast<half_t*>(init_c_buffer_data),
                           init_c_dims.production());
      auto* init_c_gpu_data = init_c_buf_t_->mutable_data(
          TARGET(kOpenCL), init_c_cpu_t->memory_size());
      TargetWrapperCL::MemcpySync(init_c_gpu_data,
                                  init_c_cpu_t->raw_data(),
                                  init_c_cpu_t->memory_size(),
                                  IoDirection::HtoD);
    } else {
      init_c_buf_t_ = new Tensor;
      const auto init_c_dims = init_c[layer_idx].dims();
      init_c_buf_t_->Resize(init_c_dims);
      auto init_c_gpu_data = init_c_buf_t_->mutable_data(
          TARGET(kOpenCL), init_c[layer_idx].memory_size());
      TargetWrapperCL::MemcpySync(init_c_gpu_data,
                                  init_c[layer_idx].raw_data(),
                                  init_c[layer_idx].memory_size(),
                                  IoDirection::HtoD);
    }

    if (fp16_support) {
      // fp16
      init_h_buf_t_ = std::unique_ptr<Tensor>(new Tensor);
      const auto init_h_dims = init_h[layer_idx].dims();
      auto* init_h_cpu = init_h[layer_idx].mutable_data<float>();
      auto init_h_cpu_t = std::unique_ptr<Tensor>(new Tensor);
      init_h_cpu_t->Resize(init_h_dims);
      auto* init_h_buffer_data = MUTABLE_DATA_CPU(init_h_cpu_t.get());
      FloatArray2HalfArray(static_cast<float*>(init_h_cpu),
                           static_cast<half_t*>(init_h_buffer_data),
                           init_h_dims.production());
      auto* init_h_gpu_data = init_h_buf_t_->mutable_data(
          TARGET(kOpenCL), init_h_cpu_t->memory_size());
      TargetWrapperCL::MemcpySync(init_h_gpu_data,
                                  init_h_cpu_t->raw_data(),
                                  init_h_cpu_t->memory_size(),
                                  IoDirection::HtoD);
    } else {
      init_h_buf_t_ = std::unique_ptr<Tensor>(new Tensor);
      auto init_h_gpu_data = init_h_buf_t_->mutable_data(
          TARGET(kOpenCL), init_h[layer_idx].memory_size());
      TargetWrapperCL::MemcpySync(init_h_gpu_data,
                                  init_h[layer_idx].raw_data(),
                                  init_h[layer_idx].memory_size(),
                                  IoDirection::HtoD);
    }

    const int& reverse_flag = is_reverse ? -1 : 1;
    bool has_allocate_mem_c = false;
    // major....
    for (int i = 0; i < time_step; i++) {
      std::cout << "===============================" << i
                << "===============================" << std::endl;
      if ("LSTM" == mode) {
        lstm_cell(gate_value,
                  &vec[1 + offset * 4],
                  init_h_holder,
                  nullptr,
                  output,
                  &vec[3 + offset * 4],
                  i,
                  state,
                  time_step);
      } else if ("GRU" == mode) {
      }

      bool in_mask = (reverse_flag * i) >= mask_min_length;
      // if (in_mask) {
      // }
    }
  }

  void PrepareForRun() override {
    rnn_param_ = param_.get_mutable<param_t>();
    auto weight_list = rnn_param_->WeightList;
    int num_layers = rnn_param_->num_layers;
    is_bidirec_ = rnn_param_->is_bidirec;
    rnn_mode_ = rnn_param_->mode;
    int gate_num = 0;
    if ("LSTM" == rnn_mode_) {
      gate_num = 4;
    } else if ("GRU" == rnn_mode_) {
      gate_num = 3;
    } else {
      LOG(FATAL) << "OpenCL RNN ERROR: unsupport mode except gru and lstm,"
                    " present mode is "
                 << rnn_mode_;
      return;
    }
    parameter_lists_.reserve(num_layers);
    reset_parameter_vector(
        weight_list, num_layers, gate_num, is_bidirec_, &parameter_lists_);

    std::vector<Tensor> vec = parameter_lists_[0];
    // weight bias int
    const auto vec0_dims = vec[0].dims();
    DDim vec0_trans_dims =
        DDim(std::vector<DDim::value_type>{vec0_dims[1], vec0_dims[0]});
    std::cout << "vec0_trans_dims: " << vec0_trans_dims[0] << "; "
              << vec0_trans_dims[1] << std::endl;
    auto vec0_cpu_tensor = std::unique_ptr<Tensor>(new Tensor);
    vec0_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    CLImageConverterFolder vec0_converter;
    const DDim& vec0_image_dims =
        vec0_converter.InitImageDimInfoWith(vec0_trans_dims);
    vec0_cpu_tensor->Resize({1, vec0_image_dims[0], vec0_image_dims[1], 4});
    auto* vec0_image_data = MUTABLE_DATA_CPU(vec0_cpu_tensor);
    auto* vec0_cpu = vec[0].mutable_data<float>();
    std::vector<float> vec0_trans_cpu(vec0_trans_dims.production());
    TransW(vec0_cpu, vec0_trans_cpu.data(), vec0_dims[0], vec0_dims[1]);
    vec0_converter.NCHWToImage(
        vec0_trans_cpu.data(), vec0_image_data, vec0_trans_dims);
    MUTABLE_DATA_GPU(
        vec0_gpu_t_, vec0_image_dims[0], vec0_image_dims[1], vec0_image_data);

    const auto vec1_dims = vec[1].dims();
    DDim vec1_trans_dims =
        DDim(std::vector<DDim::value_type>{vec1_dims[1], vec1_dims[0]});
    std::cout << "vec1_trans_dims: " << vec1_trans_dims[0] << "; "
              << vec1_trans_dims[1] << std::endl;
    auto vec1_cpu_tensor = std::unique_ptr<Tensor>(new Tensor);
    vec1_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    CLImageConverterFolder vec1_converter;
    const DDim& vec1_image_dims =
        vec1_converter.InitImageDimInfoWith(vec1_trans_dims);
    vec1_cpu_tensor->Resize({1, vec1_image_dims[0], vec1_image_dims[1], 4});
    auto* vec1_image_data = MUTABLE_DATA_CPU(vec1_cpu_tensor);
    auto* vec1_cpu = vec[1].mutable_data<float>();
    std::vector<float> vec1_trans_cpu(vec1_trans_dims.production());
    TransW(vec1_cpu, vec1_trans_cpu.data(), vec1_dims[0], vec1_dims[1]);
    vec1_converter.NCHWToImage(
        vec1_trans_cpu.data(), vec1_image_data, vec1_trans_dims);
    MUTABLE_DATA_GPU(
        vec1_gpu_t_, vec1_image_dims[0], vec1_image_dims[1], vec1_image_data);

    auto vec2_cpu_tensor = std::unique_ptr<Tensor>(new Tensor);
    vec2_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    CLImageConverterFolder vec2_converter;
    const DDim& vec2_image_dims =
        vec2_converter.InitImageDimInfoWith(vec[2].dims());
    vec2_cpu_tensor->Resize({1, vec2_image_dims[0], vec2_image_dims[1], 4});
    auto* vec2_image_data = MUTABLE_DATA_CPU(vec2_cpu_tensor);
    auto* vec2_cpu = vec[2].data<float>();
    vec2_converter.NCHWToImage(
        const_cast<float*>(vec2_cpu), vec2_image_data, vec[2].dims());
    MUTABLE_DATA_GPU(
        vec2_gpu_t_, vec2_image_dims[0], vec2_image_dims[1], vec2_image_data);

    auto vec3_cpu_tensor = std::unique_ptr<Tensor>(new Tensor);
    vec3_gpu_t_ = std::unique_ptr<Tensor>(new Tensor);
    CLImageConverterFolder vec3_converter;
    const DDim& vec3_image_dims =
        vec3_converter.InitImageDimInfoWith(vec[3].dims());
    vec3_cpu_tensor->Resize({1, vec3_image_dims[0], vec3_image_dims[1], 4});
    auto* vec3_image_data = MUTABLE_DATA_CPU(vec3_cpu_tensor);
    auto* vec3_cpu = vec[3].data<float>();
    vec3_converter.NCHWToImage(
        const_cast<float*>(vec3_cpu), vec3_image_data, vec[3].dims());
    MUTABLE_DATA_GPU(
        vec3_gpu_t_, vec3_image_dims[0], vec3_image_dims[1], vec3_image_data);
  }

  void ReInitWhenNeeded() override {
    const auto x_dims = rnn_param_->Input->dims();
    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      auto& context = ctx_->As<OpenCLContext>();

      // choose kernel
      preprocess_kernel_func_name_ = "rnn_gemm_4x4";
      context.cl_context()->AddKernel(preprocess_kernel_func_name_,
                                      "buffer/rnn_kernel.cl",
                                      build_options_,
                                      time_stamp_);
      STL::stringstream kernel_key;
      kernel_key << preprocess_kernel_func_name_ << build_options_
                 << time_stamp_;
      preprocess_kernel_ = context.cl_context()->GetKernel(kernel_key.str());

      // choose kernel
      lstm_gemm_kernel_func_name_ = "rnn_lstm_gemm";
      context.cl_context()->AddKernel(lstm_gemm_kernel_func_name_,
                                      "buffer/rnn_kernel.cl",
                                      build_options_,
                                      time_stamp_);
      STL::stringstream lstm_gemm_kernel_key;
      lstm_gemm_kernel_key << lstm_gemm_kernel_func_name_ << build_options_
                           << time_stamp_;
      lstm_gemm_kernel_ =
          context.cl_context()->GetKernel(lstm_gemm_kernel_key.str());

      // choose kernel
      lstm_compute_kernel_func_name_ = "rnn_lstm_compute";
      context.cl_context()->AddKernel(lstm_compute_kernel_func_name_,
                                      "buffer/rnn_kernel.cl",
                                      build_options_,
                                      time_stamp_);
      STL::stringstream lstm_compute_kernel_key;
      lstm_compute_kernel_key << lstm_compute_kernel_func_name_
                              << build_options_ << time_stamp_;
      lstm_compute_kernel_ =
          context.cl_context()->GetKernel(lstm_compute_kernel_key.str());

      // compute global work size
      // GetGlobalWorkSize();
    }
  }

  void GetGlobalWorkSize() {
    // if (kernel_func_name_ == "fc_gemv_1x4" || kernel_func_name_ ==
    // "adreno_gemv_1x4") {  // gemv
    //   global_work_size_ = cl::NDRange{static_cast<size_t>((n_ + 3) / 4)};
    // } else {  // gemm
    //   // local_work_size_ = cl::NDRange(32, 4, 16);
    //   global_work_size_ = cl::NDRange{static_cast<size_t>((m_ + 3) / 4),
    //                                   static_cast<size_t>((n_ + 3) / 4)};
    // }
  }

  void Run() override {
    auto input = rnn_param_->Input;
    auto output = rnn_param_->Out;
    auto pre_state = rnn_param_->PreState;
    auto state = rnn_param_->State;
    int num_layers = rnn_param_->num_layers;
    Tensor* input_holder;
    Tensor* output_holder = output;
    Tensor temp, gate_value;
    bool has_allocate_mem = false;
    const Tensor* sequence_length = rnn_param_->SequenceLength;
    std::vector<Tensor> init_h_unbind, init_c_unbind, last_h_unbind,
        last_c_unbind;
    std::vector<Tensor *> init_h_unbind_t, init_c_unbind_t, last_h_unbind_t,
        last_c_unbind_t;
    std::cout << "pre_state[0]->dims()[0]: " << pre_state[0]->dims()[0]
              << std::endl;
    init_h_unbind.resize(pre_state[0]->dims()[0]);
    last_h_unbind.resize(state[0]->dims()[0]);
    if ("LSTM" == rnn_mode_) {
      init_c_unbind.resize(pre_state[1]->dims()[0]);
      last_c_unbind.resize(state[1]->dims()[0]);
    }
    std::vector<int> stride1, stride2;
    // unbind
    for (int i = 0; i < pre_state[0]->dims()[0]; i++) {
      stride1.push_back(1);
      int dim1 = pre_state[0]->dims()[1];
      int dim2 = pre_state[0]->dims()[2];
      DDimLite dims(std::vector<int64_t>{dim1, dim2});  // 1, 512
      init_h_unbind[i].Resize(dims);
      last_h_unbind[i].Resize(dims);
      init_h_unbind_t.push_back(&init_h_unbind[i]);
      last_h_unbind_t.push_back(&last_h_unbind[i]);
      last_h_unbind[i].mutable_data<float>();
    }
    lite::host::math::split(
        pre_state[0]->data<float>(), init_h_unbind_t, 0, stride1);

    if ("LSTM" == rnn_mode_) {
      for (int i = 0; i < pre_state[1]->dims()[0]; i++) {
        stride2.push_back(1);
        int dim1 = pre_state[1]->dims()[1];
        int dim2 = pre_state[1]->dims()[2];
        DDimLite dims(std::vector<int64_t>{dim1, dim2});
        init_c_unbind[i].Resize(dims);
        last_c_unbind[i].Resize(dims);
        init_c_unbind_t.push_back(&init_c_unbind[i]);
        last_c_unbind_t.push_back(&last_c_unbind[i]);
        last_c_unbind[i].mutable_data<float>();
      }
      lite::host::math::split(
          pre_state[1]->data<float>(), init_c_unbind_t, 0, stride2);
    }
    std::vector<Tensor> output_vec(2);
    int time_step = input->dims()[0];
    int batch_size = input->dims()[1];
    int hidden_size = output->dims()[2];
    if (is_bidirec_) {
      for (int i = 0; i < 2; ++i) {
        output_vec[i].Resize({time_step, batch_size, hidden_size / 2});
        output_vec[i].mutable_data<float>();
      }
    }
    for (int i = 0; i < num_layers; i++) {
      if (i > 0) {
        if (!has_allocate_mem) {
          temp.Resize(output->dims());
          temp.mutable_data<float>();
          input_holder = &temp;
          has_allocate_mem = true;
        }
        SwapPoniter(&output_holder, &input_holder);
      }

      const Tensor* input_temp_holder = input;
      if (i > 0) {
        input_temp_holder = input_holder;
      }

      if (is_bidirec_) {
      } else {
        RunRnnLayer(input_temp_holder,
                    parameter_lists_[i],
                    init_h_unbind,
                    init_c_unbind,
                    sequence_length,
                    &last_h_unbind,
                    &last_c_unbind,
                    output_holder,
                    i,
                    &gate_value,
                    false,
                    0,
                    rnn_mode_,
                    state);
      }
    }

    std::cout << "Run end~~~" << std::endl;
  }

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = preprocess_kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 private:
  int m_, n_, k_;
  param_t* rnn_param_{nullptr};
  std::string preprocess_kernel_func_name_{};
  std::string lstm_gemm_kernel_func_name_{};
  std::string lstm_compute_kernel_func_name_{};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  std::string rnn_mode_{"LSTM"};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  bool is_adreno_{true};
  bool is_bidirec_{false};

  std::unique_ptr<Tensor> lstm_gemm_out_{nullptr};

  std::unique_ptr<Tensor> vec0_gpu_t_{nullptr};
  std::unique_ptr<Tensor> vec1_gpu_t_{nullptr};
  std::unique_ptr<Tensor> vec2_gpu_t_{nullptr};
  std::unique_ptr<Tensor> vec3_gpu_t_{nullptr};
  Tensor* init_c_buf_t_;
  Tensor* last_c_buf_t_;
  std::unique_ptr<Tensor> init_h_buf_t_;
  // std::unique_ptr<Tensor> init_c_buf_t_;
  std::vector<std::vector<Tensor>> parameter_lists_;
  cl::Buffer* init_c_buf_;
  cl::Buffer* last_c_buf_;
  cl::NDRange global_work_size_;
  cl::NDRange local_work_size_;
  cl::Kernel preprocess_kernel_;
  cl::Kernel lstm_gemm_kernel_;
  cl::Kernel lstm_compute_kernel_;
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(
    rnn, kOpenCL, kFP16, kNCHW, paddle::lite::kernels::opencl::RnnCompute, def)
    .BindInput("Input", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindInput("WeightList", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("PreState", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("SequenceLength", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("DropoutState", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Reserve", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("Out", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .BindOutput("State", {LiteType::GetTensorTy(TARGET(kOpenCL))})
    .Finalize();
